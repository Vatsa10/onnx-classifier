import * as ort from "onnxruntime-web";
import { ModelSession } from "./loadModel";
import { PreprocessedImage } from "./preprocess";
import { getWasteLabel } from "./wasteLabels";

export interface InferenceResult {
  logits: Float32Array;
  topPrediction: {
    index: number;
    label: string;
    confidence: number;
  };
  inferenceTime: number;
}

export async function runInference(
  modelSession: ModelSession,
  preprocessedImage: PreprocessedImage
): Promise<InferenceResult> {
  try {
    const startTime = performance.now();
    
    // Prepare inputs
    const feeds = { [modelSession.inputName]: preprocessedImage.tensor };
    
    // Run inference
    const results = await modelSession.session.run(feeds);
    
    // Get output
    const output = results[modelSession.outputName];
    if (!output) {
      throw new Error(`Output tensor '${modelSession.outputName}' not found`);
    }

    // Convert output to Float32Array if needed
    const logits = output.data as Float32Array;
    
    // Get top prediction
    const topPrediction = getTopPrediction(logits);
    
    const endTime = performance.now();
    const inferenceTime = endTime - startTime;

    console.log('Inference completed successfully');
    console.log('Inference time:', inferenceTime.toFixed(2), 'ms');
    console.log('Top prediction:', topPrediction);

    return {
      logits,
      topPrediction,
      inferenceTime,
    };
  } catch (error) {
    console.error('Inference failed:', error);
    throw new Error(`Inference failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

function getTopPrediction(logits: Float32Array): { index: number; label: string; confidence: number } {
  let maxIndex = 0;
  let maxValue = logits[0];

  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxValue) {
      maxValue = logits[i];
      maxIndex = i;
    }
  }

  // Convert logits to probabilities using softmax
  const probabilities = softmax(logits);
  const confidence = probabilities[maxIndex];

  return {
    index: maxIndex,
    label: getWasteLabel(maxIndex),
    confidence,
  };
}

function softmax(logits: Float32Array): Float32Array {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map(logit => Math.exp(logit - maxLogit));
  const sumExp = expLogits.reduce((sum, exp) => sum + exp, 0);
  
  return expLogits.map(exp => exp / sumExp);
}

export async function runBatchInference(
  modelSession: ModelSession,
  preprocessedImages: PreprocessedImage[]
): Promise<InferenceResult[]> {
  const results: InferenceResult[] = [];
  
  for (const image of preprocessedImages) {
    const result = await runInference(modelSession, image);
    results.push(result);
  }
  
  return results;
}

export function formatInferenceResult(result: InferenceResult): string {
  return `
Top Prediction: ${result.topPrediction.label}
Confidence: ${(result.topPrediction.confidence * 100).toFixed(2)}%
Inference Time: ${result.inferenceTime.toFixed(2)}ms
Top 5 Logits: ${Array.from(result.logits)
  .map((logit, index) => ({ index, value: logit }))
  .sort((a, b) => b.value - a.value)
  .slice(0, 5)
  .map(item => `Class ${item.index}: ${item.value.toFixed(4)}`)
  .join(', ')}
  `.trim();
}
