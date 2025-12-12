declare module 'onnxruntime-web' {
  export class Tensor {
    constructor(type: string, data: Float32Array | Int32Array | Uint8Array, dims: number[]);
    data: Float32Array | Int32Array | Uint8Array;
    dims: number[];
    type: string;
    size: number;
  }

  export interface InferenceSession {
    inputNames: string[];
    outputNames: string[];
    run(feeds: { [key: string]: Tensor }): Promise<{ [key: string]: Tensor }>;
  }

  export interface SessionOptions {
    executionProviders?: string[];
    graphOptimizationLevel?: string;
    enableCpuMemArena?: boolean;
    enableMemPattern?: boolean;
    logSeverityLevel?: number;
  }

  export const env: {
    wasm: {
      wasmPaths: string;
      numThreads: number;
      proxy?: boolean;
      fetchBinaryFile?: (path: string) => Promise<ArrayBuffer>;
    };
  };

  export type InferenceSessionCreateInput = string | ArrayBuffer | Uint8Array;

  export namespace InferenceSession {
    export function create(
      model: InferenceSessionCreateInput,
      options?: SessionOptions
    ): Promise<InferenceSession>;
  }

  export function initWasm(): Promise<void>;
}
