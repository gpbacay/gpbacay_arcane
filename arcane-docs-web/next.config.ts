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
  // The model file lives outside public/ so the download route can gate it; make sure deploys ship it.
  outputFileTracingIncludes: { "/api/arc1-download": ["./private/models/**"] },
  turbopack: {
    root: path.join(__dirname, '..'),
  },
};

export default nextConfig;
