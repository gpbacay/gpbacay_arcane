import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'miro.medium.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
        port: '',
        pathname: '/**',
      },
    ],
  },
  async redirects() {
    return [
      { source: "/docs/mnist-demo", destination: "/docs/arc-1", permanent: true },
    ];
  },
  // Parent of this app, so the docs markdown outside it can be traced into the functions.
  outputFileTracingRoot: path.join(__dirname, ".."),
  outputFileTracingIncludes: {
    "/api/arc1-download": ["./private/models/**"],
    "/docs/blog/[slug]": ["../docs/**/*.md"],
    "/docs/research": ["../docs/Closing_the_Alignment_Gap_RSAA.md"],
    "/docs/resonant-gser": ["../docs/RESONANT_GSER.md"],
    "/docs/predictive-resonant-layer": ["../docs/PREDICTIVE_RESONANT_LAYER.md"],
    "/docs/inference-time-learning": ["../docs/INFERENCE_TIME_RESONANCE.md"],
    "/docs/neural-resonance": ["../docs/NEURAL_RESONANCE.md"],
    "/docs/homeostatic-plasticity": ["../docs/HOMEOSTATIC_PLASTICITY.md"],
  },
  turbopack: {
    root: path.join(__dirname, ".."),
  },
};

export default nextConfig;
