"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import Link from "next/link";

const CANVAS_SIZE = 280;
const MODEL_SIZE = 28;
const DEBOUNCE_MS = 400; // Run inference this many ms after user stops drawing.
// Always use same-origin proxy so it works on Vercel; backend URL is set via MNIST_API_URL (server-side).
const PREDICT_URL = "/api/mnist-predict";

type Prediction = {
  digit: number;
  confidence: number;
  probabilities: number[];
} | null;

export default function MnistDemoPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [prediction, setPrediction] = useState<Prediction>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isDrawing = useRef(false);
  const hasDrawn = useRef(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const getCtx = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    return canvas.getContext("2d");
  }, []);

  const clearCanvas = useCallback(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    hasDrawn.current = false;
    const ctx = getCtx();
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setPrediction(null);
    setError(null);
  }, [getCtx]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }, []);

  const getCoords = useCallback((target: HTMLCanvasElement, clientX: number, clientY: number) => {
    const rect = target.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  }, []);

  const startDrawAt = useCallback((x: number, y: number) => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    isDrawing.current = true;
    hasDrawn.current = true;
    const ctx = getCtx();
    if (!ctx) return;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
  }, [getCtx]);

  const drawTo = useCallback((x: number, y: number) => {
    if (!isDrawing.current) return;
    const ctx = getCtx();
    if (!ctx) return;
    ctx.lineTo(x, y);
    ctx.stroke();
  }, [getCtx]);

  const endDraw = useCallback(() => {
    isDrawing.current = false;
  }, []);

  const predict = useCallback(async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    setError(null);
    setLoading(true);
    setPrediction(null);
    try {
      const offscreen = document.createElement("canvas");
      offscreen.width = MODEL_SIZE;
      offscreen.height = MODEL_SIZE;
      const ctx = offscreen.getContext("2d");
      if (!ctx) throw new Error("Could not get 2d context");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
      ctx.drawImage(canvas, 0, 0, CANVAS_SIZE, CANVAS_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE);
      const blob = await new Promise<Blob | null>((resolve) =>
        offscreen.toBlob(resolve, "image/png")
      );
      if (!blob) throw new Error("Failed to export canvas");
      const buf = await blob.arrayBuffer();
      const base64 = btoa(
        new Uint8Array(buf).reduce((acc, b) => acc + String.fromCharCode(b), "")
      );
      const res = await fetch(PREDICT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64 }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = (data && typeof data.error === "string") ? data.error : res.statusText;
        throw new Error(msg);
      }
      setPrediction({
        digit: data.digit,
        confidence: data.confidence,
        probabilities: data.probabilities ?? [],
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const schedulePredict = useCallback(() => {
    if (!hasDrawn.current) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      debounceRef.current = null;
      predict();
    }, DEBOUNCE_MS);
  }, [predict]);

  const handleEndDraw = useCallback(() => {
    endDraw();
    schedulePredict();
  }, [endDraw, schedulePredict]);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const startDraw = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = getCoords(e.target as HTMLCanvasElement, e.clientX, e.clientY);
    startDrawAt(x, y);
  }, [getCoords, startDrawAt]);

  const draw = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = getCoords(e.target as HTMLCanvasElement, e.clientX, e.clientY);
    drawTo(x, y);
  }, [getCoords, drawTo]);

  const handleTouchStart = useCallback((e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const touch = e.touches[0];
    if (!touch) return;
    const { x, y } = getCoords(e.target as HTMLCanvasElement, touch.clientX, touch.clientY);
    startDrawAt(x, y);
  }, [getCoords, startDrawAt]);

  const handleTouchMove = useCallback((e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const touch = e.touches[0];
    if (!touch) return;
    const { x, y } = getCoords(e.target as HTMLCanvasElement, touch.clientX, touch.clientY);
    drawTo(x, y);
  }, [getCoords, drawTo]);

  const handleTouchEnd = useCallback((e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    handleEndDraw();
  }, [handleEndDraw]);

  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl mb-4 text-zinc-100 leading-tight">
          MNIST Demo
        </h1>
        <p className="text-xl text-zinc-400">
          Draw a digit and run inference with the trained ARCANE MNIST model (<code className="text-[#C785F2] bg-zinc-900 px-1.5 py-0.5 rounded">mnist_arcane_model.weights.h5</code>).
        </p>
      </div>

      <div className="space-y-6 text-zinc-300">
        <h2 id="model-architecture" className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Model architecture
        </h2>
        <p className="leading-relaxed">
          The classifier treats each 28×28 image as a sequence of 28 rows. Input is normalized (Rescaling 1/255), then passed through two stacked{" "}
          <Link href="/docs/layers#predictive-resonant" className="text-[#C785F2] hover:text-[#d49cf5] underline font-medium">
            PredictiveResonantLayer
          </Link>{" "}
          blocks (128 units, local predictive resonance with optional stateful alignment), each followed by LayerNorm. A SpatioTemporalSummarization layer mixes global context, then GlobalAveragePooling1D pools over time. A BioplasticDenseLayer (128 units, with optional inference-time plasticity) and Dropout(0.2) feed into a Dense(10, softmax) output. Weights are saved as <code className="bg-zinc-900 px-1.5 py-0.5 rounded">mnist_arcane_model.weights.h5</code>.
        </p>

        <h2 id="try-it" className="text-2xl font-bold tracking-tight text-zinc-100 mt-10 mb-4 border-b border-zinc-800 pb-2">
          Try it
        </h2>
        <p>
          Draw a digit (0–9) in the box below. Inference runs automatically shortly after you stop drawing. You can also click <strong>Predict</strong> for immediate results.
        </p>

        <div className="flex flex-col sm:flex-row gap-8 items-start">
          <div className="flex flex-col gap-3">
            <canvas
              ref={canvasRef}
              width={CANVAS_SIZE}
              height={CANVAS_SIZE}
              className="border-2 border-zinc-700 bg-white cursor-crosshair touch-none rounded-none"
              style={{ imageRendering: "pixelated", touchAction: "none" }}
              onMouseDown={startDraw}
              onMouseMove={draw}
              onMouseUp={handleEndDraw}
              onMouseLeave={handleEndDraw}
              onTouchStart={handleTouchStart}
              onTouchMove={handleTouchMove}
              onTouchEnd={handleTouchEnd}
              onTouchCancel={handleTouchEnd}
            />
            <div className="flex gap-2">
              <button
                type="button"
                onClick={clearCanvas}
                className="px-4 py-2 rounded-none border border-zinc-600 bg-zinc-900 text-zinc-200 hover:bg-zinc-800 transition-colors text-sm font-medium"
              >
                Clear
              </button>
              <button
                type="button"
                onClick={predict}
                disabled={loading}
                className="px-4 py-2 rounded-none bg-[#C785F2] text-black font-semibold hover:bg-[#d49cf5] disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm"
              >
                {loading ? "Predicting…" : "Predict"}
              </button>
            </div>
          </div>

          <div className="min-w-[200px] space-y-4">
            {error && (
              <p className="text-red-400 text-sm" role="alert">
                {error}
              </p>
            )}
            <div className="p-4 rounded-none border border-zinc-700 bg-zinc-900/80">
              <p className="text-zinc-500 text-xs uppercase tracking-wider mb-2">ARCANE MNIST Classifier</p>
              <p className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Prediction</p>
              <p className="text-3xl font-bold text-[#C785F2]">
                {prediction ? prediction.digit : loading ? "..." : "—"}
              </p>
              <p className="text-zinc-400 text-sm mt-1">
                Confidence: <span className="text-white font-medium">
                  {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : loading ? "..." : "—"}
                </span>
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-zinc-500 text-xs uppercase tracking-wider">All classes</p>
              <div className="flex gap-1 flex-wrap">
                {(prediction?.probabilities ?? Array(10).fill(0)).map((p, i) => (
                  <div
                    key={i}
                    className="flex flex-col items-center gap-0.5"
                    title={prediction ? `${i}: ${(p * 100).toFixed(1)}%` : undefined}
                  >
                    <div
                      className="w-6 bg-zinc-700 rounded-sm overflow-hidden flex flex-col justify-end"
                      style={{ height: 60 }}
                    >
                      <div
                        className="w-full bg-[#C785F2] transition-all"
                        style={{ height: `${p * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-zinc-500">{i}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="rounded-none border border-amber-900/50 bg-amber-900/10 p-4 mt-6">
          <p className="text-sm text-amber-200">
            <strong>Local:</strong> Run <code className="bg-zinc-900 px-1.5 py-0.5 text-amber-100">npm run dev:with-mnist</code> from <code className="bg-zinc-900 px-1.5 py-0.5">arcane-docs-web</code> to start Next.js and the Python MNIST API together (backend URL is set automatically). Or start the API from repo root: <code className="bg-zinc-900 px-1.5 py-0.5">python examples/serve_mnist_api.py</code> and set <code className="bg-zinc-900 px-1.5 py-0.5">MNIST_API_URL=http://127.0.0.1:8000</code> in <code className="bg-zinc-900 px-1.5 py-0.5">.env.local</code>.
          </p>
          <p className="text-sm text-amber-200 mt-2">
            <strong>Vercel + Render:</strong> The app uses a same-origin proxy (<code className="bg-zinc-900 px-1.5 py-0.5">/api/mnist-predict</code>). Deploy the MNIST API to Render (repo root has <code className="bg-zinc-900 px-1.5 py-0.5">render.yaml</code>; connect the repo, deploy the web service, ensure <code className="bg-zinc-900 px-1.5 py-0.5">mnist_arcane_model.weights.h5</code> is at repo root). In Vercel, set <code className="bg-zinc-900 px-1.5 py-0.5">MNIST_API_URL</code> to your Render service URL (e.g. <code className="bg-zinc-900 px-1.5 py-0.5">https://arcane-mnist-api.onrender.com</code>). No local paths; backend URL is read from the environment at runtime.
          </p>
        </div>
      </div>
    </div>
  );
}
