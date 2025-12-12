import torch
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

# Load the model
print("Loading model...")
model = torch.load("full_mobilenetv3_tinyvit_model.pth", weights_only=False)
model.eval()

# Print model summary
print("\n" + "="*50)
print("MODEL SUMMARY")
print("="*50)
print(model)
print("\n" + "="*50)
print("MODEL DETAILS")
print("="*50)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Model input shape: [batch_size, 3, 224, 224]")
print(f"Model output shape: [batch_size, 9]")

# Test with dummy input
print("\nTesting model with dummy input...")
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (probabilities): {torch.softmax(output, dim=1)}")
    print(f"Predicted class: {torch.argmax(output, dim=1).item()}")

print("\nModel loaded and tested successfully!")
