import * as ort from "onnxruntime-web";

const MODEL_PATH = "/models/mobilenetv3_tinyvit_fused.onnx";

export async function loadModelWithExternalData(): Promise<ort.InferenceSession> {
  try {
    // Configure ONNX Runtime wasm backend to serve assets from public folder
    ort.env.wasm.wasmPaths = "/onnxruntime/";
    ort.env.wasm.numThreads = 1;

    const session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      enableCpuMemArena: true,
      enableMemPattern: true,
      logSeverityLevel: 0,
    });

    return session;
  } catch (error) {
    console.error("Failed to load model with external data:", error);
    throw error;
  }
}
