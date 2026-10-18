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
        memory?: { text: string; label: string | null; arguments?: Record<string, unknown> }[];
      },
    );
    classify(
      text: string,
      labels: string[],
    ): Promise<{ label: string | null; confidence: number; distribution: Record<string, number>; source?: string }>;
    neighbors(text: string, labels?: string[] | null): { text: string; label: string | null; score: number }[];
    remember(text: string, label: string | null, args?: Record<string, unknown> | null): void;
    toJSON(): { text: string; label: string | null; arguments?: Record<string, unknown> }[];
  }
}
