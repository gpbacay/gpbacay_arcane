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
    react(text: string, labels?: string[] | null): { votes: Record<string, number>; neighbors: { text: string; label: string | null; score: number }[] };
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

declare module "gpbacay-arcane/capsule" {
  import type { Hippocampus } from "gpbacay-arcane/hippocampus";
  export const TAG: string;
  export const MAX_PAYLOAD: number;
  export function chunk(text: string, size?: number, overlap?: number): string[];
  export function pack(documents: Record<string, string>, options?: { chunkSize?: number; overlap?: number }): Promise<Uint8Array[]>;
  export function unpack(
    payloads: (Uint8Array | null)[],
    options?: ConstructorParameters<typeof Hippocampus>[1],
  ): Promise<{ memory: Hippocampus; lost: number }>;
}
