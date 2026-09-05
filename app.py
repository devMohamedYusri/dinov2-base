#!/usr/bin/env python3
"""
FastAPI Visual Search Service for Egyptian Furniture Catalog.
Optimized for Render Free Tier (512MB RAM):
- Uses INT8 quantized DINOv2-base with in-graph L2 normalization.
- CPU memory arena disabled to release memory after each inference.
- Exposes POST /search, POST /embed, and GET /health.
"""

import io
import os
import gc
import logging
from typing import Optional, List, Dict, Any

import httpx
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
from qdrant_client import QdrantClient, models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("search-api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "furniture_catalog_small")
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "dinov2_small_quant.onnx")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(
    title="Furniture Visual Search API",
    description="Search 46,900+ Egyptian furniture catalog items by image similarity using DINOv2-small",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded single-thread ONNX Session (Memory Guard)
# ---------------------------------------------------------------------------
_session = None
_qdrant_client = None


def get_onnx_session() -> ort.InferenceSession:
    """Lazy-load the ONNX session with strict memory limits."""
    global _session
    if _session is None:
        if not os.path.exists(ONNX_MODEL_PATH):
            raise RuntimeError(
                f"Model file '{ONNX_MODEL_PATH}' not found. "
                "Ensure dinov2_small_quant.onnx is present in the project directory."
            )
        logger.info("Loading ONNX model into memory (arena disabled for 512MB RAM)...")
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False  # Immediately release memory after each inference
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _session = ort.InferenceSession(
            ONNX_MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        logger.info("ONNX model loaded successfully.")
    return _session


def get_qdrant() -> QdrantClient:
    """Lazy-load Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        if not QDRANT_URL:
            logger.warning("QDRANT_URL is not set. Vector search will fail.")
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


# ---------------------------------------------------------------------------
# Image Preprocessing & Inference
# ---------------------------------------------------------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Convert PIL image to normalized CHW tensor shape (1, 3, 224, 224)."""
    # Downscale in PIL first to conserve memory
    img = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.ascontiguousarray(arr[np.newaxis, :, :, :], dtype=np.float32)


def generate_embedding(session: ort.InferenceSession, img: Image.Image) -> List[float]:
    """Run single image inference and return L2-normalized vector."""
    input_tensor = preprocess_image(img)
    # Dynamically detect input tensor name (e.g. 'pixel_values', 'x', 'arg0')
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    raw_vec = outputs[0]

    # Handle both wrapped 2D output (1, dim) and raw 3D output (1, seq_len, dim)
    if raw_vec.ndim == 3:
        cls_token = raw_vec[:, 0, :]
        norm = np.linalg.norm(cls_token, axis=1, keepdims=True)
        vec = (cls_token / (norm + 1e-12)).squeeze()
    else:
        vec = raw_vec.squeeze()

    # Free memory
    del input_tensor
    del outputs
    gc.collect()

    return vec.tolist()


async def load_image_from_bytes_or_url(
    file: Optional[UploadFile] = None, image_url: Optional[str] = None
) -> Image.Image:
    """Load and validate an image from either a file upload or an image URL."""
    if file and hasattr(file, "filename") and file.filename:
        content = await file.read()
        if content:
            try:
                return Image.open(io.BytesIO(content))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    if image_url and image_url.strip():
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(image_url.strip())
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download image from URL (status {resp.status_code})",
                    )
                return Image.open(io.BytesIO(resp.content))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error fetching image_url: {str(e)}")

    raise HTTPException(status_code=400, detail="Must provide either 'file' upload or 'image_url'.")


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class SearchUrlRequest(BaseModel):
    image_url: str
    limit: int = 10
    category: Optional[str] = None
    store_name: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None


class SearchResultItem(BaseModel):
    id: str
    score: float
    title: str
    price_egp: float
    category: Optional[str] = None
    store_name: Optional[str] = None
    product_url: str
    image_url: str
    in_stock: bool = True
    payload: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    total_results: int
    results: List[SearchResultItem]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Service info and documentation links."""
    return {
        "service": "Egyptian Furniture Visual Search API",
        "status": "online",
        "model": "DINOv2-small (INT8 quantized ONNX, 384-dim)",
        "catalog": f"Collection '{COLLECTION_NAME}' (46,900+ Egyptian furniture products)",
        "documentation": "/docs",
        "endpoints": {
            "search_multipart": "POST /search (file upload or image_url form field)",
            "search_json": "POST /search-json (JSON body: {image_url, limit, category, store_name})",
            "embed": "POST /embed (generates 384-dim normalized embedding)",
            "health": "GET /health (service uptime check)",
        },
    }


@app.get("/health")
def health():
    """Health check for Render uptime monitors."""
    model_exists = os.path.exists(ONNX_MODEL_PATH)
    return {
        "status": "healthy",
        "model_file_exists": model_exists,
        "model_path": ONNX_MODEL_PATH,
        "qdrant_url_configured": bool(QDRANT_URL),
        "collection_name": COLLECTION_NAME,
    }


@app.post("/embed")
async def embed(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
):
    """
    Generate and return the 384-dim normalized embedding for an image.
    Accepts either multipart file upload or 'image_url' form field.
    """
    try:
        session = get_onnx_session()
        img = await load_image_from_bytes_or_url(file, image_url)
        try:
            embedding = generate_embedding(session, img)
        finally:
            img.close()
        return {"dim": len(embedding), "embedding": embedding}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /embed")
        raise HTTPException(status_code=500, detail=f"Embed error: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_by_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    limit: int = Form(10),
    category: Optional[str] = Form(None),
    store_name: Optional[str] = Form(None),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None),
):
    """
    Visual Search endpoint (Multipart Form).
    Accepts:
    - file: Uploaded image file (e.g., from user's camera or device)
    - image_url: Public image URL
    - limit: Number of top results to return (default: 10)
    - category: Filter by category (e.g. 'sofa', 'chair', 'bed', 'table')
    - store_name: Filter by store ('Chic Homz', 'Hub Furniture', 'Homzmart')
    - min_price / max_price: Price range in EGP
    """
    try:
        session = get_onnx_session()
        qdrant = get_qdrant()

        img = await load_image_from_bytes_or_url(file, image_url)
        try:
            query_vector = generate_embedding(session, img)
        finally:
            img.close()

        # Build Qdrant filters if requested
        must_conditions = []
        if category:
            must_conditions.append(
                models.FieldCondition(key="category", match=models.MatchValue(value=category.lower()))
            )
        if store_name:
            must_conditions.append(
                models.FieldCondition(key="store_name", match=models.MatchValue(value=store_name))
            )
        if min_price is not None or max_price is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="price_egp",
                    range=models.Range(
                        gte=min_price if min_price is not None else 0.0,
                        lte=max_price if max_price is not None else float("inf"),
                    ),
                )
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        try:
            search_results = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Qdrant query failed: {e}")
            raise HTTPException(status_code=500, detail=f"Qdrant search error: {str(e)}")

        formatted_results = []
        for p in search_results.points:
            payload = p.payload or {}
            raw_price = payload.get("price_egp")
            try:
                price_val = float(raw_price) if raw_price is not None else 0.0
            except (ValueError, TypeError):
                price_val = 0.0

            formatted_results.append(
                SearchResultItem(
                    id=str(p.id),
                    score=float(p.score),
                    title=str(payload.get("title", "Unknown")),
                    price_egp=price_val,
                    category=payload.get("category"),
                    store_name=payload.get("store_name"),
                    product_url=str(payload.get("product_url", "")),
                    image_url=str(payload.get("image_url", "")),
                    in_stock=bool(payload.get("in_stock", True)),
                    payload=payload,
                )
            )

        return SearchResponse(total_results=len(formatted_results), results=formatted_results)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /search")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/search-json", response_model=SearchResponse)
async def search_by_image_json(payload: SearchUrlRequest):
    """
    Visual Search endpoint (JSON body).
    Convenient for backend-to-backend calls where image URL is passed via JSON.
    """
    return await search_by_image(
        file=None,
        image_url=payload.image_url,
        limit=payload.limit,
        category=payload.category,
        store_name=payload.store_name,
        min_price=payload.min_price,
        max_price=payload.max_price,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
