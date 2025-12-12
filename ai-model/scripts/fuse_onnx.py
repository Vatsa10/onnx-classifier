import onnx
from onnx.external_data_helper import convert_model_from_external_data
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "mobilenetv3_tinyvit.onnx"
OUTPUT_PATH = ROOT / "onnx-classifier" / "public" / "models" / "mobilenetv3_tinyvit_fused.onnx"

print(f"Loading model from {MODEL_PATH}...")
model = onnx.load(MODEL_PATH, load_external_data=True)
print("Converting model to embed external data...")
convert_model_from_external_data(model)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, OUTPUT_PATH)
print(f"Saved fused model to {OUTPUT_PATH}")
