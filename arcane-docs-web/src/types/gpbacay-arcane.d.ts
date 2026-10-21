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
        memory?: { text: string; label: string | null; arguments?: Record<string, unknown>; source?: "auto" }[];
        maxSize?: number | null;
        confident?: number;
        threshold?: number;
        leak?: number;
        maxPending?: number;
        teacher?: ((text: string, labels: string[]) => Promise<string | Record<string, number> | null>) | null;
      },
    );
    classify(
      text: string,
      labels: string[],
    ): Promise<{ label: string | null; confidence: number; distribution: Record<string, number>; source?: string }>;
    neighbors(text: string, labels?: string[] | null): { text: string; label: string | null; score: number }[];
    remember(text: string, label: string | null, args?: Record<string, unknown> | null): string;
    observe(
      text: string,
      label: string | null,
      args?: Record<string, unknown> | null,
      trust?: number,
      decisionLabels?: string[] | null,
      opts?: Record<string, unknown>,
    ): Promise<string | null>;
    toJSON(): { text: string; label: string | null; arguments?: Record<string, unknown>; source?: "auto" }[];
  }
}
