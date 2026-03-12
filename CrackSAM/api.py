import os
import io
import asyncio
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response

from inference import load_model, predict_image_tiled

logger = logging.getLogger("cracksam.api")

# Only one request hits the GPU at a time; others queue.
_gpu_semaphore = asyncio.Semaphore(1)


def _build_model_config() -> SimpleNamespace:
    """Build a config namespace from environment variables (with defaults)."""
    return SimpleNamespace(
        img_size=int(os.getenv("CRACKSAM_IMG_SIZE", "448")),
        num_classes=int(os.getenv("CRACKSAM_NUM_CLASSES", "1")),
        ckpt=os.getenv("CRACKSAM_CKPT", "checkpoints/sam_vit_h_4b8939.pth"),
        delta_ckpt=os.getenv("CRACKSAM_DELTA_CKPT", "checkpoints/CrackSAM_adapter_d32.pth"),
        vit_name=os.getenv("CRACKSAM_VIT_NAME", "vit_h"),
        delta_type=os.getenv("CRACKSAM_DELTA_TYPE", "adapter"),
        middle_dim=int(os.getenv("CRACKSAM_MIDDLE_DIM", "32")),
        scaling_factor=float(os.getenv("CRACKSAM_SCALING_FACTOR", "0.2")),
        rank=int(os.getenv("CRACKSAM_RANK", "4")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = _build_model_config()
    logger.info("Loading model (vit=%s, delta=%s)...", cfg.vit_name, cfg.delta_type)
    net = load_model(cfg)
    logger.info("Model loaded.")
    app.state.net = net
    app.state.cfg = cfg
    yield


app = FastAPI(title="CrackSAM", lifespan=lifespan)


@app.get("/health")
async def health():
    model_loaded = hasattr(app.state, "net") and app.state.net is not None
    return {"status": "ok" if model_loaded else "unavailable", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    overlap: int = Query(64, ge=0),
    batch_size: int = Query(8, ge=1),
):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(img) / 255.0

    cfg = app.state.cfg
    multimask_output = cfg.num_classes > 1
    loop = asyncio.get_running_loop()

    async with _gpu_semaphore:
        prediction = await loop.run_in_executor(
            None,
            predict_image_tiled,
            app.state.net,
            image_np,
            cfg.img_size,
            multimask_output,
            overlap,
            batch_size,
        )

    mask = (prediction * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
