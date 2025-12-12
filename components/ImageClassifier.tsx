'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { checkServerStatus, predictWithServer, ServerInferenceResult } from '../utils/serverInference';
import { createImagePreview } from '../utils/preprocess';

interface ImageClassifierState {
  isLoading: boolean;
  isServerReady: boolean;
  isProcessing: boolean;
  selectedImage: string | null;
  result: ServerInferenceResult | null;
  error: string | null;
  serverStatus: string | null;
  cameraActive: boolean;
}

export default function ImageClassifier() {
  const [state, setState] = useState<ImageClassifierState>({
    isLoading: false,
    isServerReady: false,
    isProcessing: false,
    selectedImage: null,
    result: null,
    error: null,
    serverStatus: null,
    cameraActive: false,
  });

  // Check server connection on component mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        setState(prev => ({ ...prev, isLoading: true, error: null }));
        
        const status = await checkServerStatus();
        
        setState(prev => ({
          ...prev,
          isLoading: false,
          isServerReady: status.model_loaded,
          serverStatus: status.device,
        }));
        
        console.log('Server connection established:', status);
      } catch (error) {
        console.error('Server connection failed:', error);
        setState(prev => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : 'Failed to connect to server',
        }));
      }
    };

    checkServer();
  }, []);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const handleImageUpload = useCallback(async (file: File) => {
    if (!state.isServerReady) {
      setState(prev => ({ ...prev, error: 'Server not ready yet' }));
      return;
    }

    try {
      setState(prev => ({ ...prev, isProcessing: true, error: null, result: null }));

      // Create image preview
      const imageUrl = await createImagePreview(file);
      setState(prev => ({ ...prev, selectedImage: imageUrl }));

      // Run server inference
      const result = await predictWithServer(imageUrl);

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
  }, [state.isServerReady]);

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
    if (!videoRef.current || !state.isServerReady) {
      setState(prev => ({ ...prev, error: 'Camera not ready or server not ready' }));
      return;
    }

    try {
      setState(prev => ({ ...prev, isProcessing: true, error: null, result: null }));

      // Capture image from video
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (ctx && videoRef.current) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
        const imageUrl = canvas.toDataURL();
        setState(prev => ({ ...prev, selectedImage: imageUrl }));

        // Run server inference
        const result = await predictWithServer(imageUrl);

        setState(prev => ({
          ...prev,
          isProcessing: false,
          result,
        }));
      }
    } catch (error) {
      console.error('Camera capture failed:', error);
      setState(prev => ({
        ...prev,
        isProcessing: false,
        error: error instanceof Error ? error.message : 'Failed to capture from camera',
      }));
    }
  }, [state.isServerReady]);

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
          state.isServerReady ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
        }`}>
          Server: {state.isServerReady ? 'Ready' : 'Connecting...'}
        </div>
        {state.serverStatus && (
          <div className="px-3 py-1 rounded-full text-sm bg-blue-100 text-blue-800">
            Device: {state.serverStatus}
          </div>
        )}
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
          disabled={!state.isServerReady || state.isProcessing}
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
            disabled={!state.isServerReady}
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
            {state.result.class_name}
          </div>
        </div>
      )}
    </div>
  );
}
