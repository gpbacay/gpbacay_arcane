#!/usr/bin/env node
/**
 * Starts the Arcane docs site with the MNIST inference API (Python FastAPI)
 * as a child process — microservices-style, single command.
 *
 * Usage: npm run dev:with-mnist   (from arcane-docs-web)
 *        node scripts/start-with-mnist.js
 *
 * Requires: Python on PATH, and from repo root: pip install fastapi uvicorn Pillow
 *           Trained weights at repo root: mnist_arcane_model.weights.h5
 */

const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const MNIST_PORT = 8000;
const MNIST_HEALTH_URL = `http://127.0.0.1:${MNIST_PORT}/health`;
const POLL_MS = 400;
const POLL_TIMEOUT_MS = 30000;

const repoRoot = path.resolve(__dirname, "..", "..");
const pythonScript = path.join(repoRoot, "examples", "serve_mnist_api.py");

function waitForHealth() {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    function poll() {
      const req = http.get(MNIST_HEALTH_URL, (res) => {
        if (res.statusCode === 200) return resolve();
        if (Date.now() < deadline) setTimeout(poll, POLL_MS);
        else reject(new Error("MNIST API health check timed out"));
      });
      req.on("error", () => {
        if (Date.now() >= deadline) reject(new Error("MNIST API did not become ready"));
        else setTimeout(poll, POLL_MS);
      });
    }
    poll();
  });
}

function main() {
  // Prefer "python" (Windows and many Linux/Mac); set MNIST_PYTHON_CMD to override e.g. python3
  const pythonCmd = process.env.MNIST_PYTHON_CMD || (process.platform === "win32" ? "python" : "python3");
  console.log("[mnist] Starting Python MNIST API at", pythonScript);
  // On Windows with shell: true, paths with spaces (e.g. "project test") get split;
  // run as one quoted command so Python receives the full path.
  const useShell = process.platform === "win32";
  const py = useShell
    ? spawn(
        `${pythonCmd} "${pythonScript.replace(/"/g, '""')}"`,
        { cwd: repoRoot, stdio: "inherit", shell: true, env: { ...process.env, PYTHONUNBUFFERED: "1" } }
      )
    : spawn(pythonCmd, [pythonScript], {
        cwd: repoRoot,
        stdio: "inherit",
        shell: false,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

  py.on("error", (err) => {
    console.error("[mnist] Failed to start Python:", err.message);
    console.error("[mnist] Ensure Python is on PATH and run from repo root: pip install fastapi uvicorn Pillow");
    process.exit(1);
  });

  py.on("exit", (code, signal) => {
    if (code !== null && code !== 0) {
      console.error("[mnist] Python API exited with code", code);
    }
    if (nextRef.current) nextRef.current.kill(signal || "SIGTERM");
    process.exit(code ?? 1);
  });

  const nextRef = { current: null };

  waitForHealth()
    .then(() => {
      console.log("[mnist] MNIST API ready at http://localhost:" + MNIST_PORT);
      console.log("[next]  Starting Next.js dev server...");
      const next = spawn("npm", ["run", "dev"], {
        cwd: path.resolve(__dirname, ".."),
        stdio: "inherit",
        shell: true,
        env: { ...process.env, MNIST_API_URL: "http://127.0.0.1:" + MNIST_PORT },
      });
      nextRef.current = next;
      next.on("exit", (code, signal) => {
        py.kill(signal || "SIGTERM");
        process.exit(code ?? 0);
      });
    })
    .catch((err) => {
      console.error("[mnist]", err.message);
      py.kill("SIGTERM");
      process.exit(1);
    });

  process.on("SIGINT", () => {
    py.kill("SIGINT");
    if (nextRef.current) nextRef.current.kill("SIGINT");
    process.exit(0);
  });
  process.on("SIGTERM", () => {
    py.kill("SIGTERM");
    if (nextRef.current) nextRef.current.kill("SIGTERM");
    process.exit(0);
  });
}

main();
