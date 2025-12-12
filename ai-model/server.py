import pickle
import sys
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals, safe_globals
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import MobileNetV3
import timm
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import base64
import io
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = Path(__file__).parent / "full_mobilenetv3_tinyvit_model.pth"

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=9, cnn_out_dim=576, vit_out_dim=448):
        super().__init__()
        self.cnn = mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.DEFAULT")
        self.cnn.classifier = nn.Identity()
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        self.vit = timm.create_model("tiny_vit_11m_224", pretrained=True)
        if hasattr(self.vit, "head"):
            self.vit.head = nn.Identity()
        elif hasattr(self.vit, "fc"):
            self.vit.fc = nn.Identity()
        self.vit_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(cnn_out_dim + vit_out_dim, num_classes)

    def forward(self, x):
        cnn_feat_map = self.cnn.features(x)
        pooled = self.cnn_pool(cnn_feat_map).view(x.size(0), -1)

        vit_feat_raw = self.vit(x)
        vit_feat = self.vit_pool(vit_feat_raw).view(x.size(0), -1)

        fused = torch.cat([pooled, vit_feat], dim=1)
        return self.fc(fused)


HybridCNNTransformer.__module__ = "__main__"
setattr(sys.modules.setdefault("__main__", sys.modules[__name__]), "HybridCNNTransformer", HybridCNNTransformer)


class InferenceRequest(BaseModel):
    image: str  # Base64 encoded image


class InferenceResponse(BaseModel):
    class_name: str
    confidence: float
    top_predictions: List[Dict[str, Any]]


# Global variables
model = None
device = None
preprocess = None


def load_model_global():
    """Load the PyTorch model globally"""
    global model, device, preprocess
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = HybridCNNTransformer()
    
    # Load checkpoint with safe globals
    allowed_globals = [HybridCNNTransformer, MobileNetV3]
    add_safe_globals(allowed_globals)
    
    try:
        with safe_globals(allowed_globals):
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        with safe_globals(allowed_globals):
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Extract state dict
    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(checkpoint)}")
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Initialize preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    logger.info("Model loaded successfully")


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@torch.no_grad()
def predict_image(image: Image.Image) -> Dict[str, Any]:
    """Run inference on a single image"""
    global model, device, preprocess
    
    # Preprocess image
    tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Run inference
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    
    # Get top prediction
    top_prob, top_idx = torch.max(probs, dim=1)
    top_class = CLASS_NAMES[top_idx.item()]
    top_confidence = top_prob.item()
    
    # Get top-5 predictions
    topk_probs, topk_indices = torch.topk(probs, k=min(5, len(CLASS_NAMES)), dim=1)
    
    top_predictions = []
    for prob, idx in zip(topk_probs[0], topk_indices[0]):
        top_predictions.append({
            "class_name": CLASS_NAMES[idx.item()],
            "confidence": prob.item()
        })
    
    return {
        "class_name": top_class,
        "confidence": top_confidence,
        "top_predictions": top_predictions
    }


# Initialize FastAPI app
app = FastAPI(
    title="Waste Classification API",
    description="PyTorch model server for waste classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    
    load_model_global()
    logger.info("Server startup complete")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }


@app.get("/classes")
async def get_classes():
    """Get available class names"""
    return {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES)
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Predict waste class from base64 encoded image"""
    try:
        # Decode image
        image = decode_base64_image(request.image)
        
        # Run inference
        result = predict_image(image)
        
        return InferenceResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
