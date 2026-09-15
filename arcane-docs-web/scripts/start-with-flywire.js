#!/usr/bin/env node
/**
 * Starts the Arcane docs site with the FlyWire connectome API (Python FastAPI).
 *
 * Usage: npm run dev:with-flywire   (from arcane-docs-web)
 *
 * Requires: Python on PATH, and from repo root:
 *   pip install -r requirements-flywire.txt
 */

const { spawn } = require("child_process");
const path = require("path");
const http = require("http");

const FLYWIRE_PORT = 8001;
const FLYWIRE_HEALTH_URL = `http://127.0.0.1:${FLYWIRE_PORT}/health`;
const POLL_MS = 400;
const POLL_TIMEOUT_MS = 30000;

const repoRoot = path.resolve(__dirname, "..", "..");
const pythonScript = path.join(repoRoot, "examples", "serve_flywire_api.py");

function waitForHealth() {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    function poll() {
      const req = http.get(FLYWIRE_HEALTH_URL, (res) => {
        if (res.statusCode === 200) return resolve();
        if (Date.now() < deadline) setTimeout(poll, POLL_MS);
        else reject(new Error("FlyWire API health check timed out"));
      });
      req.on("error", () => {
        if (Date.now() >= deadline) reject(new Error("FlyWire API did not become ready"));
        else setTimeout(poll, POLL_MS);
      });
    }
    poll();
  });
}

function main() {
  const pythonCmd = process.env.FLYWIRE_PYTHON_CMD || process.env.MNIST_PYTHON_CMD || (process.platform === "win32" ? "python" : "python3");
  console.log("[flywire] Starting Python FlyWire API at", pythonScript);
  const useShell = process.platform === "win32";
  const py = useShell
    ? spawn(
        `${pythonCmd} "${pythonScript.replace(/"/g, '""')}"`,
        { cwd: repoRoot, stdio: "inherit", shell: true, env: { ...process.env, PYTHONUNBUFFERED: "1", PORT: String(FLYWIRE_PORT) } }
      )
    : spawn(pythonCmd, [pythonScript], {
        cwd: repoRoot,
        stdio: "inherit",
        shell: false,
        env: { ...process.env, PYTHONUNBUFFERED: "1", PORT: String(FLYWIRE_PORT) },
      });

  py.on("error", (err) => {
    console.error("[flywire] Failed to start Python:", err.message);
    console.error("[flywire] From repo root: pip install -r requirements-flywire.txt");
    process.exit(1);
  });

  py.on("exit", (code, signal) => {
    if (code !== null && code !== 0) {
      console.error("[flywire] Python API exited with code", code);
    }
    if (nextRef.current) nextRef.current.kill(signal || "SIGTERM");
    process.exit(code ?? 1);
  });

  const nextRef = { current: null };

  waitForHealth()
    .then(() => {
      console.log("[flywire] API ready at http://localhost:" + FLYWIRE_PORT);
      console.log("[next]  Starting Next.js dev server...");
      const next = spawn("npm", ["run", "dev"], {
        cwd: path.resolve(__dirname, ".."),
        stdio: "inherit",
        shell: true,
        env: { ...process.env, FLYWIRE_API_URL: "http://127.0.0.1:" + FLYWIRE_PORT },
      });
      nextRef.current = next;
      next.on("exit", (code, signal) => {
        py.kill(signal || "SIGTERM");
        process.exit(code ?? 0);
      });
    })
    .catch((err) => {
      console.error("[flywire]", err.message);
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
