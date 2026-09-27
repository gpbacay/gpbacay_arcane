"use strict";
// ARC 1 for Node.js. Runs the Python model as a child process and talks to it
// over stdin/stdout (see gpbacay_arcane/arc1_stdio.py). Needs: pip install gpbacay-arcane
const { spawn } = require("node:child_process");
const readline = require("node:readline");

/**
 * Start ARC 1 and resolve once the model is loaded.
 * @param {{ python?: string, model?: string }} [options]
 *   python: interpreter with gpbacay-arcane installed (default $ARC1_PYTHON, then python/python3)
 *   model:  path to an .rcn file (default: the bundled arc1-tiny)
 */
function load(options = {}) {
  const python = options.python || process.env.ARC1_PYTHON || (process.platform === "win32" ? "python" : "python3");
  const args = ["-m", "gpbacay_arcane.arc1_stdio", ...(options.model ? ["--model", options.model] : [])];
  const proc = spawn(python, args, { stdio: ["pipe", "pipe", "pipe"] });
  const pending = new Map();
  let nextId = 1;
  let stderr = "";
  let exited = null;

  proc.stderr.on("data", (d) => {
    stderr = (stderr + d).slice(-4000); // keep the tail for error messages
  });

  return new Promise((resolve, reject) => {
    const fail = (err) => {
      exited = err;
      for (const { reject: r } of pending.values()) r(err);
      pending.clear();
      reject(err); // no-op once ready
    };
    proc.on("error", (e) => fail(new Error(`could not start ${python}: ${e.message}`)));
    proc.on("exit", (code) => fail(new Error(`ARC 1 process exited (code ${code})\n${stderr}`)));

    readline.createInterface({ input: proc.stdout }).on("line", (line) => {
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        return;
      }
      if (msg.ready) return resolve(agent);
      const p = pending.get(msg.id);
      if (!p) return;
      pending.delete(msg.id);
      msg.error ? p.reject(new Error(msg.error)) : p.resolve(msg.result);
    });

    const call = (method, params) => {
      if (exited) return Promise.reject(exited);
      const id = nextId++;
      return new Promise((res, rej) => {
        pending.set(id, { resolve: res, reject: rej });
        proc.stdin.write(JSON.stringify({ id, method, params }) + "\n");
      });
    };

    const agent = {
      /** Pick a tool and fill its arguments. tools: [{ name, description, parameters: [{ name, type, description, required, enum }] }] */
      run: (prompt, tools, opts = {}) => call("run", { prompt, tools, ...opts }),
      /** Pull typed fields out of text. schema: { field: { type, description, enum? } } or { field: "description" } */
      extract: (text, schema, opts = {}) => call("extract", { text, schema, ...opts }),
      /** Pick one label. opts: { task?, descriptions?, cycles? } */
      classify: (text, labels, opts = {}) => call("classify", { text, labels, ...opts }),
      /** L2-normalised embedding vector. */
      embed: (text) => call("embed", { text }),
      /** Stop the Python process. */
      close: () => {
        proc.removeAllListeners("exit");
        proc.stdin.end();
        proc.kill();
      },
    };
  });
}

module.exports = { load };
