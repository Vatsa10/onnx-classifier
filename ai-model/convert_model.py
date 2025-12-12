import torch
import torchvision.transforms as transforms
from torch.onnx import export
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import timm

# Define the HybridCNNTransformer class (must match the original)
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=9, cnn_out_dim=576, vit_out_dim=448):
        super().__init__()
        # MobileNetV3 small backbone
        self.cnn = mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
        self.cnn.classifier = nn.Identity()
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        
        # TinyViT from timm
        self.vit = timm.create_model('tiny_vit_11m_224', pretrained=True)
        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()
        elif hasattr(self.vit, 'fc'):
            self.vit.fc = nn.Identity()
        self.vit_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final fused classifier
        self.fc = nn.Linear(cnn_out_dim + vit_out_dim, num_classes)

    def forward(self, x):
        # CNN path
        cnn_feat_map = self.cnn.features(x)
        pooled = self.cnn_pool(cnn_feat_map).view(x.size(0), -1)
        
        # ViT path
        vit_feat_raw = self.vit(x)
        vit_feat = self.vit_pool(vit_feat_raw).view(x.size(0), -1)
        
        # Concatenate and classify
        fused = torch.cat([pooled, vit_feat], dim=1)
        out = self.fc(fused)
        return out

# Load PyTorch model
model = torch.load("full_mobilenetv3_tinyvit_model.pth", weights_only=False)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(model, dummy_input, "mobilenetv3_tinyvit.onnx", 
                  export_params=True, opset_version=18,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})