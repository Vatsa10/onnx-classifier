import pickle
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals, safe_globals
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import MobileNetV3
import timm
from PIL import Image
from pathlib import Path
from typing import List

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "full_mobilenetv3_tinyvit_model.pth"
IMAGE_PATHS = [
    PROJECT_ROOT / "WhatsApp Image 2025-12-11 at 15.27.02_bacfca02.jpg",
    PROJECT_ROOT / "WhatsApp Image 2025-12-11 at 15.27.03_56424892.jpg",
    PROJECT_ROOT / "WhatsApp Image 2025-12-11 at 15.27.04_31cb8c92.jpg",
]

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


def load_model(device: torch.device) -> nn.Module:
    model = HybridCNNTransformer()
    allowed_globals = [HybridCNNTransformer, MobileNetV3]
    add_safe_globals(allowed_globals)
    try:
        with safe_globals(allowed_globals):
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def run_inference(model: nn.Module, image_paths: List[Path], device: torch.device):
    preprocess = build_transform()
    for path in image_paths:
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            continue

        image = Image.open(path).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        top_prob, top_idx = torch.max(probs, dim=1)
        label = CLASS_NAMES[top_idx.item()]

        print(f"\nImage: {path.name}")
        print(f"Top prediction: {label} ({top_prob.item() * 100:.2f}%)")

        topk_probs, topk_indices = torch.topk(probs, k=min(5, len(CLASS_NAMES)), dim=1)
        print("Top-5:")
        for rank, (prob, idx) in enumerate(zip(topk_probs[0], topk_indices[0]), start=1):
            print(f"  {rank}. {CLASS_NAMES[idx]} - {prob.item() * 100:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(device)
    run_inference(model, IMAGE_PATHS, device)


if __name__ == "__main__":
    main()
