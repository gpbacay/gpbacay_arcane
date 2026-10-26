/// <reference lib="webworker" />
// SmolVLM-256M-Instruct Q4. Its text decoder is SmolLM2-135M. The vision encoder is stubbed, so this is text only.
// Same browser path as the portfolio chatbot (Transformers.js, WebGPU, one worker thread).
import {
  env,
  AutoProcessor,
  AutoModelForImageTextToText,
  TextStreamer,
  InterruptableStoppingCriteria,
  type ProgressInfo,
} from "@huggingface/transformers";

env.allowLocalModels = false;
env.useBrowserCache = true;
if (env.backends.onnx.wasm) env.backends.onnx.wasm.numThreads = 1;

const MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct";
const Q4 = { embed_tokens: "q4", vision_encoder: "q4", decoder_model_merged: "q4" } as const;
const Q4F16 = { embed_tokens: "q4f16", vision_encoder: "q4f16", decoder_model_merged: "q4f16" } as const;

type InMsg =
  | { type: "load" }
  | { type: "generate"; id: number; prompt: string; maxNewTokens: number }
  | { type: "interrupt" };

type Processor = Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>;
type Model = Awaited<ReturnType<typeof AutoModelForImageTextToText.from_pretrained>>;

let processor: Processor | null = null;
let model: Model | null = null;
let stopping: InterruptableStoppingCriteria | null = null;

const post = (msg: object) => self.postMessage(msg);

// ponytail: this class always opens a vision session. A 65-byte Identity graph stands in so SigLIP is never downloaded.
const VISION_STUB = new Uint8Array([
  8, 8, 58, 55, 10, 16, 10, 1, 120, 18, 1, 121, 34, 8, 73, 100, 101, 110, 116, 105, 116, 121, 18, 1, 103, 90, 15, 10,
  1, 120, 18, 10, 10, 8, 8, 1, 18, 4, 10, 2, 8, 1, 98, 15, 10, 1, 121, 18, 10, 10, 8, 8, 1, 18, 4, 10, 2, 8, 1, 66, 4,
  10, 0, 16, 13,
]);

function cacheName(key: RequestInfo): string {
  return typeof key === "string" ? key : key.url;
}

async function skipVisionDownload() {
  const browser = await caches.open("transformers-cache");
  for (const req of await browser.keys()) {
    if (req.url.includes("vision_encoder")) await browser.delete(req);
  }
  env.useCustomCache = true;
  env.useBrowserCache = false;
  env.customCache = {
    match(key: RequestInfo) {
      if (cacheName(key).includes("vision_encoder")) return Promise.resolve(new Response(VISION_STUB.slice()));
      return browser.match(key);
    },
    put(key: RequestInfo, response: Response) {
      if (cacheName(key).includes("vision_encoder")) return Promise.resolve();
      return browser.put(key, response);
    },
  };
}

const files = new Map<string, number>();
function onProgress(info: ProgressInfo) {
  if ("file" in info && info.file.includes("vision_encoder")) return;
  if (info.status === "progress") {
    files.set(info.file, info.progress / 100);
    const vals = [...files.values()];
    post({ type: "status", phase: "downloading", progress: vals.reduce((a, b) => a + b, 0) / vals.length });
  } else if (info.status === "done") {
    post({ type: "status", phase: "loading", progress: 1 });
  }
}

async function loadModel(dtype: typeof Q4 | typeof Q4F16) {
  processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: onProgress });
  model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
    dtype,
    device: "webgpu",
    progress_callback: onProgress,
  });
}

async function load() {
  try {
    await skipVisionDownload();
    try {
      await loadModel(Q4);
    } catch (e) {
      console.warn("[SmolLM2] q4 files unavailable, loading q4f16:", e);
      files.clear();
      await loadModel(Q4F16);
    }
    const vision = model?.sessions.vision_encoder as { release?: () => Promise<void> } | undefined;
    await vision?.release?.();
    if (model) delete model.sessions.vision_encoder;
    post({ type: "status", phase: "warming", progress: 1 });
    await run("What is ARC 1?\nFacts: ARC 1 is a System 1 decision model.", 2, 0, true);
    post({ type: "ready" });
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e) });
  }
}

async function run(prompt: string, maxNewTokens: number, id: number, quiet: boolean) {
  const proc = processor;
  const mdl = model;
  if (!proc || !mdl) throw new Error("SmolLM2 is not loaded");
  stopping?.reset();
  stopping = new InterruptableStoppingCriteria();
  const messages = [{ role: "user", content: [{ type: "text", text: prompt }] }] as never;
  const text = proc.apply_chat_template(messages, { add_generation_prompt: true, tokenize: false }) as string;
  const tokenizer = proc.tokenizer;
  if (!tokenizer) throw new Error("SmolLM2 tokenizer missing");
  const inputs = tokenizer(text, { add_special_tokens: false });
  let out = "";
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      out += chunk;
      if (!quiet) post({ type: "token", id, text: out });
    },
  });
  await mdl.generate({
    ...inputs,
    max_new_tokens: maxNewTokens,
    do_sample: false,
    repetition_penalty: 1.15,
    streamer,
    stopping_criteria: stopping,
  });
  if (!quiet) post({ type: "done", id, text: out.trim() });
}

self.onmessage = (e: MessageEvent<InMsg>) => {
  const msg = e.data;
  if (msg.type === "load") {
    void load();
    return;
  }
  if (msg.type === "interrupt") {
    stopping?.interrupt();
    return;
  }
  void run(msg.prompt, msg.maxNewTokens, msg.id, false).catch((err) => {
    post({ type: "gen-error", id: msg.id, message: err instanceof Error ? err.message : String(err) });
  });
};
