"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  applyGust,
  createBody,
  stepBody,
  stepBrain,
  type BrainSnapshot,
  type ControllerKind,
  type FlyBody,
  type LocomotionMode,
} from "@/lib/fruitfly-brain";

const WORLD_GROUND = 18;

function drawCompoundEye(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  rx: number,
  ry: number,
  highlight: number
) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(rx, ry);
  const grad = ctx.createRadialGradient(-0.25, -0.3, 0.1, 0, 0, 1);
  grad.addColorStop(0, "#ff6b6b");
  grad.addColorStop(0.45, "#c41e3a");
  grad.addColorStop(1, "#4a0a12");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(0, 0, 1, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 180, 180, 0.18)";
  ctx.lineWidth = 0.08;
  for (let i = -2; i <= 2; i++) {
    ctx.beginPath();
    ctx.ellipse(0, 0, 1, 0.22 + Math.abs(i) * 0.12, i * 0.35, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.fillStyle = `rgba(255,255,255,${0.22 + highlight * 0.15})`;
  ctx.beginPath();
  ctx.ellipse(-0.35, -0.38, 0.22, 0.14, -0.4, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawLeg(
  ctx: CanvasRenderingContext2D,
  hipX: number,
  hipY: number,
  side: number,
  phase: number,
  tucked: number
) {
  const swing = Math.sin(phase);
  const stance = Math.cos(phase);
  const reach = (22 - tucked * 10) * side;
  const lift = stance > 0 ? 4 : 14 - tucked * 8;
  const midX = hipX + reach * 0.45 + swing * 6 * side;
  const midY = hipY + 10 + (1 - tucked) * 4;
  const footX = hipX + reach + swing * 10;
  const footY = hipY + lift + 18 * (1 - tucked * 0.7);

  ctx.strokeStyle = "#a1a1aa";
  ctx.lineWidth = 1.7;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(hipX, hipY);
  ctx.lineTo(midX, midY);
  ctx.lineTo(footX, footY);
  ctx.stroke();

  ctx.fillStyle = "#d4d4d8";
  ctx.beginPath();
  ctx.arc(footX, footY, 1.6, 0, Math.PI * 2);
  ctx.fill();
}

function drawFly(ctx: CanvasRenderingContext2D, body: FlyBody, mode: LocomotionMode, wingAmp: number) {
  ctx.save();
  ctx.rotate(body.pitch);
  ctx.transform(1, 0, body.roll * 0.22, 1, 0, 0);

  const tucked = mode === "fly" ? 0.72 : 0;
  const flap = Math.sin(body.wingPhase) * wingAmp;

  const drawWing = (side: number) => {
    ctx.save();
    ctx.translate(6, -8);
    ctx.rotate(side * (0.55 + flap * 0.9));
    ctx.scale(side, 1);
    ctx.fillStyle = "rgba(199, 133, 242, 0.28)";
    ctx.strokeStyle = "rgba(185, 223, 224, 0.75)";
    ctx.lineWidth = 1.05;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.bezierCurveTo(18, -16, 54, -18, 68, 6);
    ctx.bezierCurveTo(48, 18, 18, 12, 0, 0);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(28, -4, 62, 4);
    ctx.stroke();
    ctx.restore();
  };

  drawWing(-1);
  drawWing(1);

  const legPhase = body.gaitPhase;
  const hips = [
    { x: 16, y: 6, side: -1, off: 0 },
    { x: 2, y: 8, side: -1, off: Math.PI },
    { x: -14, y: 7, side: -1, off: 0 },
    { x: 16, y: 6, side: 1, off: Math.PI },
    { x: 2, y: 8, side: 1, off: 0 },
    { x: -14, y: 7, side: 1, off: Math.PI },
  ];
  for (const hip of hips) {
    drawLeg(ctx, hip.x, hip.y, hip.side, legPhase + hip.off, tucked);
  }

  ctx.save();
  ctx.translate(-8, 2);
  ctx.rotate(Math.sin(body.wingPhase * 1.6) * 0.5);
  ctx.strokeStyle = "#f294c0";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(-10, -8);
  ctx.stroke();
  ctx.fillStyle = "#c785f2";
  ctx.beginPath();
  ctx.arc(-11, -9, 2.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  const abdomen = ctx.createLinearGradient(-8, 0, -48, 8);
  abdomen.addColorStop(0, "#d6b48a");
  abdomen.addColorStop(1, "#6b4423");
  ctx.fillStyle = abdomen;
  ctx.beginPath();
  ctx.ellipse(-28, 2, 26, 11, -0.12, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(40, 20, 10, 0.35)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    ctx.beginPath();
    ctx.ellipse(-16 - i * 6.5, 2 + i * 0.4, 7 - i * 0.7, 9 - i * 0.8, -0.12, 0.2, Math.PI - 0.2);
    ctx.stroke();
  }

  const thorax = ctx.createRadialGradient(4, -4, 2, 2, 0, 20);
  thorax.addColorStop(0, "#e8d0a8");
  thorax.addColorStop(1, "#8a6230");
  ctx.fillStyle = thorax;
  ctx.beginPath();
  ctx.ellipse(4, -1, 18, 13, 0.05, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#c4a574";
  ctx.beginPath();
  ctx.ellipse(24, -2, 12, 10, 0.1, 0, Math.PI * 2);
  ctx.fill();

  drawCompoundEye(ctx, 28, -6, 7.5, 6.2, mode === "fly" ? 1 : 0.4);
  drawCompoundEye(ctx, 22, 2, 5.2, 4.4, 0.2);

  ctx.strokeStyle = "#e4e4e7";
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.moveTo(32, -8);
  ctx.quadraticCurveTo(42, -22, 38, -28);
  ctx.moveTo(30, -10);
  ctx.quadraticCurveTo(36, -24, 30, -30);
  ctx.stroke();
  ctx.fillStyle = "#f4f4f5";
  ctx.beginPath();
  ctx.arc(38, -28, 1.6, 0, Math.PI * 2);
  ctx.arc(30, -30, 1.6, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawWorld(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  body: FlyBody,
  mode: LocomotionMode,
  brain: BrainSnapshot,
  time: number
) {
  const scale = Math.min(3.15, Math.max(2.2, w / 280));
  const flySX = w * 0.38;
  const flySY = h * 0.46;
  const groundSY = flySY + 34 * scale + (body.y - WORLD_GROUND) * 1.15;
  const scroll = body.x * 0.9;

  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, "#050508");
  sky.addColorStop(0.55, "#0b0614");
  sky.addColorStop(1, "#12081c");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  ctx.fillStyle = "rgba(199,133,242,0.08)";
  for (let i = 0; i < 22; i++) {
    const px = ((i * 97 + time * 18) % (w + 40)) - 20;
    const py = 16 + ((i * 53) % (h * 0.5));
    ctx.beginPath();
    ctx.arc(px, py, 1.2, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#0c0c10";
  ctx.fillRect(0, Math.min(h, groundSY), w, h);
  ctx.strokeStyle = "rgba(199, 133, 242, 0.22)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, groundSY);
  ctx.lineTo(w, groundSY);
  ctx.stroke();

  ctx.strokeStyle = "rgba(199, 133, 242, 0.1)";
  ctx.lineWidth = 1;
  for (let i = -8; i < 16; i++) {
    const gx = i * 56 - (scroll % 56);
    ctx.beginPath();
    ctx.moveTo(gx, groundSY);
    ctx.lineTo(gx - 18, h);
    ctx.stroke();
  }

  ctx.save();
  ctx.globalAlpha = 0.35;
  ctx.fillStyle = "#000";
  ctx.beginPath();
  ctx.ellipse(flySX, groundSY - 3, 36, 7, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  if (body.gust > 0.08) {
    ctx.strokeStyle = `rgba(157, 228, 250, ${0.18 + body.gust * 0.4})`;
    ctx.lineWidth = 1.6;
    for (let i = 0; i < 8; i++) {
      const gy = flySY - 70 + i * 18;
      const gx0 = flySX - 90 * body.gustDir;
      ctx.beginPath();
      ctx.moveTo(gx0, gy);
      ctx.bezierCurveTo(
        gx0 + 28 * body.gustDir,
        gy - 8,
        gx0 + 56 * body.gustDir,
        gy + 10,
        gx0 + 100 * body.gustDir,
        gy
      );
      ctx.stroke();
    }
  }

  ctx.save();
  ctx.translate(flySX, flySY);
  ctx.scale(scale, scale);
  ctx.shadowColor = "rgba(199, 133, 242, 0.35)";
  ctx.shadowBlur = 18;
  drawFly(ctx, body, mode, brain.motor.wingAmp);
  ctx.restore();

  const hudX = w - 198;
  const hudY = 16;
  ctx.fillStyle = "rgba(9,9,11,0.78)";
  ctx.fillRect(hudX, hudY, 182, 196);
  ctx.strokeStyle = "rgba(199,133,242,0.35)";
  ctx.strokeRect(hudX + 0.5, hudY + 0.5, 181, 195);

  ctx.fillStyle = "#a1a1aa";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("RSAA BRAIN", hudX + 12, hudY + 18);
  ctx.fillStyle = "#C785F2";
  ctx.fillText(`${brain.cycles} cycle${brain.cycles === 1 ? "" : "s"}`, hudX + 118, hudY + 18);

  brain.layers.forEach((layer, li) => {
    const y = hudY + 34 + li * 38;
    ctx.fillStyle = "#e4e4e7";
    ctx.font = "11px ui-sans-serif, system-ui";
    ctx.fillText(layer.name, hudX + 12, y);
    ctx.fillStyle = "#71717a";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(`Δ ${layer.divergence.toFixed(3)}`, hudX + 118, y);
    layer.state.forEach((v, ni) => {
      const bx = hudX + 12 + ni * 26;
      const bh = Math.max(2, Math.abs(v) * 14);
      ctx.fillStyle = v >= 0 ? "#C785F2" : "#9DE4FA";
      ctx.fillRect(bx, y + 18 - bh, 18, bh);
      ctx.fillStyle = "#27272a";
      ctx.fillRect(bx, y + 18, 18, 2);
    });
  });
}

export function FruitFlySimulation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bodyRef = useRef<FlyBody>(createBody());
  const brainRef = useRef<BrainSnapshot | null>(null);
  const modeRef = useRef<LocomotionMode>("walk");
  const kindRef = useRef<ControllerKind>("resonant");
  const runningRef = useRef(true);
  const lastRef = useRef(0);
  const timeRef = useRef(0);
  const statsTickRef = useRef(0);

  const [mode, setMode] = useState<LocomotionMode>("walk");
  const [kind, setKind] = useState<ControllerKind>("resonant");
  const [running, setRunning] = useState(true);
  const [stats, setStats] = useState({
    divergence: 0,
    altitude: 18,
    pitch: 0,
    cycles: 8,
  });

  const targetY = useCallback((m: LocomotionMode) => (m === "fly" ? 92 : WORLD_GROUND), []);

  useEffect(() => {
    modeRef.current = mode;
    kindRef.current = kind;
    runningRef.current = running;
  }, [mode, kind, running]);

  const gust = useCallback(() => {
    applyGust(bodyRef.current, 1);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frame = 0;
    const loop = (now: number) => {
      frame = requestAnimationFrame(loop);
      if (!lastRef.current) lastRef.current = now;
      const dt = Math.min(0.033, (now - lastRef.current) / 1000);
      lastRef.current = now;

      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(320, Math.floor(rect.width));
      const h = Math.max(280, Math.floor(rect.height));
      if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
        canvas.width = Math.floor(w * dpr);
        canvas.height = Math.floor(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const body = bodyRef.current;
      const m = modeRef.current;
      const k = kindRef.current;
      if (runningRef.current) {
        timeRef.current += dt;
        const brain = stepBrain(body, m, k, targetY(m));
        stepBody(body, brain.motor, m, targetY(m), dt);
        brainRef.current = brain;
      }

      const brain = brainRef.current ?? stepBrain(body, m, k, targetY(m));
      drawWorld(ctx, w, h, body, m, brain, timeRef.current);

      statsTickRef.current += dt;
      if (statsTickRef.current > 0.12) {
        statsTickRef.current = 0;
        setStats({
          divergence: brain.meanDivergence,
          altitude: body.y,
          pitch: body.pitch,
          cycles: brain.cycles,
        });
      }
    };

    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [targetY]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        gust();
      } else if (e.key === "w" || e.key === "W") {
        setMode("walk");
      } else if (e.key === "f" || e.key === "F") {
        setMode("fly");
      } else if (e.key === "r" || e.key === "R") {
        setKind((prev) => (prev === "resonant" ? "feedforward" : "resonant"));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [gust]);

  return (
    <div className="not-prose space-y-4">
      <div className="border border-zinc-800 bg-zinc-950">
        <canvas
          ref={canvasRef}
          className="block w-full h-[400px] sm:h-[460px]"
          aria-label="Virtual fruit fly driven by an ARCANE-style resonant controller"
        />
        <div className="flex flex-col gap-3 border-t border-zinc-800 p-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setMode("walk")}
              className={`px-3 py-1.5 text-xs font-semibold uppercase tracking-wider transition-colors ${
                mode === "walk"
                  ? "bg-[#C785F2] text-black"
                  : "border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
              }`}
            >
              Walk
            </button>
            <button
              type="button"
              onClick={() => setMode("fly")}
              className={`px-3 py-1.5 text-xs font-semibold uppercase tracking-wider transition-colors ${
                mode === "fly"
                  ? "bg-[#C785F2] text-black"
                  : "border border-zinc-700 bg-zinc-900 text-zinc-300 hover:bg-zinc-800"
              }`}
            >
              Fly
            </button>
            <button
              type="button"
              onClick={() => setKind((prev) => (prev === "resonant" ? "feedforward" : "resonant"))}
              className="border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-200 hover:bg-zinc-800"
            >
              {kind === "resonant" ? "Resonant RSAA" : "Feed-forward"}
            </button>
            <button
              type="button"
              onClick={gust}
              className="border border-[#9DE4FA]/40 bg-[#9DE4FA]/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-wider text-[#9DE4FA] hover:bg-[#9DE4FA]/20"
            >
              Gust
            </button>
            <button
              type="button"
              onClick={() => setRunning((v) => !v)}
              className="border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-300 hover:bg-zinc-800"
            >
              {running ? "Pause" : "Resume"}
            </button>
          </div>
          <p className="text-[11px] font-mono text-zinc-500">
            Space gust · W walk · F fly · R controller
          </p>
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="border border-zinc-800 bg-zinc-900/50 p-3">
          <dt className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">Divergence</dt>
          <dd suppressHydrationWarning className="mt-1 font-mono text-lg text-[#C785F2]">{stats.divergence.toFixed(3)}</dd>
        </div>
        <div className="border border-zinc-800 bg-zinc-900/50 p-3">
          <dt className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">Altitude</dt>
          <dd suppressHydrationWarning className="mt-1 font-mono text-lg text-zinc-100">{stats.altitude.toFixed(1)}</dd>
        </div>
        <div className="border border-zinc-800 bg-zinc-900/50 p-3">
          <dt className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">Pitch</dt>
          <dd suppressHydrationWarning className="mt-1 font-mono text-lg text-zinc-100">{(stats.pitch * 57.3).toFixed(1)}°</dd>
        </div>
        <div className="border border-zinc-800 bg-zinc-900/50 p-3">
          <dt className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">Think cycles</dt>
          <dd suppressHydrationWarning className="mt-1 font-mono text-lg text-zinc-100">{stats.cycles}</dd>
        </div>
      </dl>
    </div>
  );
}
