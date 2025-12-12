import * as ort from "onnxruntime-web";

export interface PreprocessedImage {
  tensor: ort.Tensor;
  originalImage: HTMLImageElement;
}

export async function preprocessImage(imageElement: HTMLImageElement): Promise<PreprocessedImage> {
  try {
    // Create canvas for image processing
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Could not get 2D context from canvas');
    }

    // Set canvas size to 224x224
    const targetSize = 224;
    canvas.width = targetSize;
    canvas.height = targetSize;

    // Draw and resize image
    ctx.drawImage(imageElement, 0, 0, targetSize, targetSize);

    // Get image data
    const imageData = ctx.getImageData(0, 0, targetSize, targetSize);
    const data = imageData.data; // RGBA format

    // Convert to RGB, normalize to [0, 1], and store in CHW layout expected by ONNX model
    const floatData = new Float32Array(3 * targetSize * targetSize);
    for (let y = 0; y < targetSize; y++) {
      for (let x = 0; x < targetSize; x++) {
        const pixelIndex = (y * targetSize + x) * 4;
        const r = data[pixelIndex] / 255.0;
        const g = data[pixelIndex + 1] / 255.0;
        const b = data[pixelIndex + 2] / 255.0;

        const idx = y * targetSize + x;
        floatData[idx] = r; // channel 0
        floatData[targetSize * targetSize + idx] = g; // channel 1
        floatData[2 * targetSize * targetSize + idx] = b; // channel 2
      }
    }

    // Create tensor with shape [1, 3, 224, 224] (NCHW format)
    const tensor = new ort.Tensor('float32', floatData, [1, 3, targetSize, targetSize]);

    console.log('Image preprocessed successfully');
    console.log('Tensor shape:', tensor.dims);
    console.log('Tensor data type:', tensor.type);

    return {
      tensor,
      originalImage: imageElement
    };
  } catch (error) {
    console.error('Image preprocessing failed:', error);
    throw new Error(`Preprocessing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

export async function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to load image'));
    };
    
    img.src = url;
  });
}

export async function loadImageFromVideo(videoElement: HTMLVideoElement): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        reject(new Error('Could not get 2D context from canvas'));
        return;
      }

      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      ctx.drawImage(videoElement, 0, 0);

      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to create image from video'));
      img.src = canvas.toDataURL();
    } catch (error) {
      reject(error);
    }
  });
}

export function createImagePreview(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      if (e.target?.result) {
        resolve(e.target.result as string);
      } else {
        reject(new Error('Failed to read file'));
      }
    };
    
    reader.onerror = () => reject(new Error('File reading failed'));
    reader.readAsDataURL(file);
  });
}
