declare module "next-pwa" {
  import type { NextConfig } from "next";

  type WithPWAOptions = {
    dest?: string;
    register?: boolean;
    skipWaiting?: boolean;
    disable?: boolean;
  };

  export default function withPWA(options?: WithPWAOptions): (nextConfig: NextConfig) => NextConfig;
}
