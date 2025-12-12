import * as ort from "onnxruntime-web";
import { loadModelWithExternalData } from "./modelLoader";

export interface ModelSession {
  session: ort.InferenceSession;
  inputName: string;
  outputName: string;
}

export async function loadModel(): Promise<ModelSession> {
  try {
    // Load the model with external data
    const session = await loadModelWithExternalData();

    // Get input and output names
    const inputNames = session.inputNames;
    const outputNames = session.outputNames;
    
    if (inputNames.length === 0 || outputNames.length === 0) {
      throw new Error('Model has no inputs or outputs');
    }

    console.log('Model loaded successfully');
    console.log('Input names:', inputNames);
    console.log('Output names:', outputNames);
    return {
      session,
      inputName: inputNames[0],
      outputName: outputNames[0],
    };
  } catch (error) {
    console.error('Failed to load ONNX model:', error);
    throw new Error(`Model loading failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function checkWebGPUSupport(): Promise<boolean> {
  try {
    if (!(navigator as any).gpu) {
      console.log('WebGPU not supported by browser');
      return false;
    }

    const adapter = await (navigator as any).gpu.requestAdapter();
    if (!adapter) {
      console.log('No WebGPU adapter found');
      return false;
    }

    console.log('WebGPU supported:', adapter);
    return true;
  } catch (error) {
    console.log('WebGPU check failed:', error);
    return false;
  }
}
