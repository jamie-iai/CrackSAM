import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.ndimage import zoom
from PIL import Image
from importlib import import_module
from segment_anything import sam_model_registry


SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def load_model(args):
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    if args.delta_type == 'adapter':
        pkg = import_module('delta.sam_adapter_image_encoder')
        net = pkg.Adapter_Sam(sam, args.middle_dim, args.scaling_factor).cuda()
    elif args.delta_type == 'lora':
        pkg = import_module('delta.sam_lora_image_encoder')
        net = pkg.LoRA_Sam(sam, args.rank).cuda()
    else:
        pkg = import_module('delta.sam_adapter_lora_image_encoder')
        net = pkg.LoRA_Adapter_Sam(sam, args.middle_dim, args.rank).cuda()

    net.load_delta_parameters(args.delta_ckpt)
    net.eval()
    return net


def predict_patch(net, patch_np, img_size, multimask_output):
    """Run inference on a single 448x448 patch (H,W,3 numpy array, values in [0,1]).
    Returns a (H,W) prediction array with values 0 or 1."""
    h, w = patch_np.shape[:2]
    image = np.transpose(patch_np, (2, 0, 1))[np.newaxis, ...]
    if h != img_size or w != img_size:
        image = zoom(image, (1, 1, img_size / h, img_size / w), order=3)

    inputs = torch.from_numpy(image).float().cuda()
    with torch.no_grad():
        outputs = net(inputs, multimask_output, img_size)
        output_masks = outputs['masks']
        prediction = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        prediction = prediction.cpu().detach().numpy()

    if h != img_size or w != img_size:
        prediction = zoom(prediction, (h / img_size, w / img_size), order=0)

    return prediction


def predict_image_resize(net, image_np, img_size, multimask_output):
    """Resize entire image to img_size, predict, resize mask back."""
    return predict_patch(net, image_np, img_size, multimask_output)


def predict_batch(net, patches_np, img_size, multimask_output):
    """Run inference on a batch of patches (N,H,W,3 numpy array, values in [0,1]).
    Returns (N,H,W) prediction array with values 0 or 1."""
    n, h, w, _ = patches_np.shape
    # (N, H, W, 3) -> (N, 3, H, W)
    images = np.transpose(patches_np, (0, 3, 1, 2))
    if h != img_size or w != img_size:
        images = zoom(images, (1, 1, img_size / h, img_size / w), order=3)

    inputs = torch.from_numpy(images).float().cuda()
    with torch.no_grad():
        outputs = net(inputs, multimask_output, img_size)
        output_masks = outputs['masks']
        predictions = torch.argmax(torch.softmax(output_masks, dim=1), dim=1)
        predictions = predictions.cpu().detach().numpy()

    if h != img_size or w != img_size:
        predictions = zoom(predictions, (1, h / img_size, w / img_size), order=0)

    return predictions


def tile_starts(length, tile_size, stride_size):
    """Compute tile start positions that fully cover [0, length).

    When length <= tile_size, returns [0] (the single tile is padded externally).
    Otherwise, returns sorted starts where every start + tile_size <= length,
    and the last start is positioned so the tile ends exactly at length.
    """
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size, stride_size))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def predict_image_tiled(net, image_np, img_size, multimask_output, overlap, batch_size=8):
    """Tile the image into overlapping patches, predict in batches, stitch together.

    In overlap regions, a pixel is marked as crack if ANY overlapping patch
    predicted it as crack (union/max strategy).
    """
    h, w = image_np.shape[:2]
    tile = img_size
    stride = max(1, tile - overlap)

    y_starts = tile_starts(h, tile, stride)
    x_starts = tile_starts(w, tile, stride)

    # Pad image if smaller than tile in either dimension
    pad_h = max(tile - h, 0)
    pad_w = max(tile - w, 0)
    if pad_h > 0 or pad_w > 0:
        image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    prediction = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

    # Collect all tile coordinates and patches
    tile_coords = [(y0, x0) for y0 in y_starts for x0 in x_starts]
    total_patches = len(tile_coords)
    logging.info(f'  Tiling: {len(y_starts)}x{len(x_starts)} = {total_patches} patches '
                 f'(tile={tile}, overlap={overlap}, batch_size={batch_size}, image={w}x{h})')

    # Process in batches
    for i in range(0, total_patches, batch_size):
        batch_coords = tile_coords[i:i+batch_size]
        patches = np.stack([image_np[y0:y0+tile, x0:x0+tile] for y0, x0 in batch_coords])
        preds = predict_batch(net, patches, img_size, multimask_output)
        for (y0, x0), pred in zip(batch_coords, preds):
            prediction[y0:y0+tile, x0:x0+tile] = np.maximum(
                prediction[y0:y0+tile, x0:x0+tile], pred
            )

    # Crop back to original size (in case we padded)
    prediction = prediction[:h, :w]
    return prediction


def main():
    parser = argparse.ArgumentParser(description='CrackSAM Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to a single image or a directory of images')
    parser.add_argument('--output_dir', type=str, default='./output/inference',
                        help='Directory to save predicted masks')
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_h_4b8939.pth',
                        help='Pretrained SAM checkpoint')
    parser.add_argument('--delta_ckpt', type=str, default='checkpoints/CrackSAM_adapter_d32.pth',
                        help='Trained delta checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_h')
    parser.add_argument('--delta_type', type=str, default='adapter',
                        choices=['adapter', 'lora', 'both'])
    parser.add_argument('--middle_dim', type=int, default=32)
    parser.add_argument('--scaling_factor', type=float, default=0.2)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3407)
    # Tiling options
    parser.add_argument('--tile', action='store_true',
                        help='Enable tiled inference for large images')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap in pixels between adjacent tiles (default: 64)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of tiles per batch during tiled inference (default: 8)')
    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Collect image paths
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = sorted([
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ])
    else:
        print(f'Error: {args.input} is not a valid file or directory')
        sys.exit(1)

    if not image_paths:
        print(f'No images found in {args.input}')
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    mode = 'tiled' if args.tile else 'resize'
    logging.info(f'Loading model...')
    multimask_output = args.num_classes > 1
    net = load_model(args)
    logging.info(f'Model loaded. Running {mode} inference on {len(image_paths)} image(s)...')

    for img_path in image_paths:
        image_np = np.array(Image.open(img_path).convert('RGB')) / 255.0

        if args.tile:
            prediction = predict_image_tiled(
                net, image_np, args.img_size, multimask_output, args.overlap,
                args.batch_size)
        else:
            prediction = predict_image_resize(
                net, image_np, args.img_size, multimask_output)

        # Save prediction as binary mask (0 or 255)
        mask = (prediction * 255).astype(np.uint8)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output_dir, f'{basename}_mask.png')
        Image.fromarray(mask).save(out_path)
        logging.info(f'Saved: {out_path}')

    logging.info(f'Done. {len(image_paths)} mask(s) saved to {args.output_dir}')


if __name__ == '__main__':
    main()
