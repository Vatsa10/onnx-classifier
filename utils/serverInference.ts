// Server-side inference utilities

export interface ServerInferenceResult {
  class_name: string;
  confidence: number;
  top_predictions: Array<{
    class_name: string;
    confidence: number;
  }>;
}

export interface ServerStatus {
  status: string;
  model_loaded: boolean;
  device: string | null;
}

export interface ServerClasses {
  classes: string[];
  num_classes: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function checkServerStatus(): Promise<ServerStatus> {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    throw new Error(`Failed to connect to server: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function getServerClasses(): Promise<ServerClasses> {
  try {
    const response = await fetch(`${API_BASE_URL}/classes`);
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    throw new Error(`Failed to get classes: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function predictWithServer(imageDataUrl: string): Promise<ServerInferenceResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: imageDataUrl // Send full data URL, server will handle parsing
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server returned ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    throw new Error(`Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}
