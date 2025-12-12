'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { loadModel, checkWebGPUSupport, ModelSession } from '../utils/loadModel';
import { preprocessImage, loadImageFromFile, loadImageFromVideo, createImagePreview } from '../utils/preprocess';
import { runInference, InferenceResult } from '../utils/runInference';

interface ImageClassifierState {
  isLoading: boolean;
  isModelLoaded: boolean;
  isProcessing: boolean;
  selectedImage: string | null;
  result: InferenceResult | null;
  error: string | null;
  webGPUSupported: boolean;
  cameraActive: boolean;
}

export default function ImageClassifier() {
  const [state, setState] = useState<ImageClassifierState>({
    isLoading: false,
    isModelLoaded: false,
    isProcessing: false,
    selectedImage: null,
    result: null,
    error: null,
    webGPUSupported: false,
    cameraActive: false,
  });

  const [modelSession, setModelSession] = useState<ModelSession | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Load model on component mount
  useEffect(() => {
    const initializeModel = async () => {
      try {
        setState(prev => ({ ...prev, isLoading: true, error: null }));
        
        // Check WebGPU support
        const webGPU = await checkWebGPUSupport();
        
        // Load the model
        const session = await loadModel();
        
        setModelSession(session);
        setState(prev => ({
          ...prev,
          isLoading: false,
          isModelLoaded: true,
          webGPUSupported: webGPU,
        }));
        
        console.log('Model initialized successfully');
      } catch (error) {
        console.error('Model initialization failed:', error);
        setState(prev => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : 'Failed to load model',
        }));
      }
    };

    initializeModel();
  }, []);

  const handleImageUpload = useCallback(async (file: File) => {
    if (!modelSession) {
      setState(prev => ({ ...prev, error: 'Model not loaded yet' }));
      return;
    }

    try {
      setState(prev => ({ ...prev, isProcessing: true, error: null, result: null }));

      // Create image preview
      const imageUrl = await createImagePreview(file);
      setState(prev => ({ ...prev, selectedImage: imageUrl }));

      // Load and preprocess image
      const imageElement = await loadImageFromFile(file);
      const preprocessed = await preprocessImage(imageElement);

      // Run inference
      const result = await runInference(modelSession, preprocessed);

      setState(prev => ({
        ...prev,
        isProcessing: false,
        result,
      }));
    } catch (error) {
      console.error('Image processing failed:', error);
      setState(prev => ({
        ...prev,
        isProcessing: false,
        error: error instanceof Error ? error.message : 'Failed to process image',
      }));
    }
  }, [modelSession]);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleImageUpload(file);
    }
  }, [handleImageUpload]);

  const startCamera = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, error: null }));
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setState(prev => ({ ...prev, cameraActive: true }));
      }
    } catch (error) {
      console.error('Camera access failed:', error);
      setState(prev => ({
        ...prev,
        error: 'Failed to access camera. Please ensure camera permissions are granted.',
      }));
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setState(prev => ({ ...prev, cameraActive: false }));
  }, []);

  const captureFromCamera = useCallback(async () => {
    if (!videoRef.current || !modelSession) {
      setState(prev => ({ ...prev, error: 'Camera not ready or model not loaded' }));
      return;
    }

    try {
      setState(prev => ({ ...prev, isProcessing: true, error: null, result: null }));

      // Capture image from video
      const imageElement = await loadImageFromVideo(videoRef.current);
      const preprocessed = await preprocessImage(imageElement);

      // Create preview
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
        const imageUrl = canvas.toDataURL();
        setState(prev => ({ ...prev, selectedImage: imageUrl }));
      }

      // Run inference
      const result = await runInference(modelSession, preprocessed);

      setState(prev => ({
        ...prev,
        isProcessing: false,
        result,
      }));
    } catch (error) {
      console.error('Camera capture failed:', error);
      setState(prev => ({
        ...prev,
        isProcessing: false,
        error: error instanceof Error ? error.message : 'Failed to capture from camera',
      }));
    }
  }, [modelSession]);

  return (
    <div className="max-w-4xl mx-auto p-4 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Waste Classification</h1>
        <p className="text-gray-600">
          Upload an image or use your camera to classify waste materials
        </p>
      </div>

      {/* Status indicators */}
      <div className="mb-4 flex flex-wrap gap-2">
        <div className={`px-3 py-1 rounded-full text-sm ${
          state.isModelLoaded ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
        }`}>
          Model: {state.isModelLoaded ? 'Loaded' : 'Loading...'}
        </div>
        <div className={`px-3 py-1 rounded-full text-sm ${
          state.webGPUSupported ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
        }`}>
          {state.webGPUSupported ? 'WebGPU' : 'WASM'}
        </div>
      </div>

      {/* Error display */}
      {state.error && (
        <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
          {state.error}
        </div>
      )}

      {/* Control buttons */}
      <div className="mb-6 flex flex-wrap gap-4">
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={!state.isModelLoaded || state.isProcessing}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          Upload Image
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        {!state.cameraActive ? (
          <button
            onClick={startCamera}
            disabled={!state.isModelLoaded}
            className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            Start Camera
          </button>
        ) : (
          <div className="flex gap-2">
            <button
              onClick={captureFromCamera}
              disabled={state.isProcessing}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              Capture
            </button>
            <button
              onClick={stopCamera}
              className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Stop Camera
            </button>
          </div>
        )}
      </div>

      {/* Camera view */}
      {state.cameraActive && (
        <div className="mb-6">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full max-w-md mx-auto rounded-lg shadow-md"
          />
        </div>
      )}

      {/* Image preview */}
      {state.selectedImage && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Selected Image</h3>
          <img
            src={state.selectedImage}
            alt="Selected for classification"
            className="w-full max-w-md mx-auto rounded-lg shadow-md"
          />
        </div>
      )}

      {/* Processing indicator */}
      {state.isProcessing && (
        <div className="mb-6 text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="mt-2 text-gray-600">Processing image...</p>
        </div>
      )}

      {/* Results */}
      {state.result && (
        <div className="mb-6 p-6 bg-gray-50 rounded-lg text-center">
          <h3 className="text-xl font-semibold text-gray-700 mb-2">Classification Result</h3>
          <div className="text-3xl font-bold text-blue-600">
            {state.result.topPrediction.label}
          </div>
        </div>
      )}
    </div>
  );
}
