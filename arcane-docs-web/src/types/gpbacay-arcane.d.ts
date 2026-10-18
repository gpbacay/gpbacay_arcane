declare module "gpbacay-arcane/web" {
  export function load(options?: {
    model?: unknown;
    config?: unknown;
    wasmPaths?: string;
  }): Promise<{
    classify(text: string, labels: string[]): Promise<{ distribution: Record<string, number> }>;
  }>;
}

declare module "gpbacay-arcane/hippocampus" {
  export class Hippocampus {
    constructor(
      model?: {
        classify(text: string, labels: string[]): Promise<{ distribution: Record<string, number> }>;
      } | null,
      options?: {
        k?: number;
        weight?: number;
        memory?: { text: string; label: string }[];
      },
    );
    classify(
      text: string,
      labels: string[],
    ): Promise<{ label: string | null; confidence: number; distribution: Record<string, number> }>;
  }
}
