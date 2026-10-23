// Occipital as a video: each payload of occipital.mjs becomes one QR code frame of an H.264 MP4, so a knowledge
// base is one file that any video tool can copy, stream or archive, and that reads back into a search. Node only.
// Needs the optional peers qrcode and jsqr, and FFmpeg: $FFMPEG_PATH, else ffmpeg-static if installed, else ffmpeg.
import { spawn } from "node:child_process";
import { statSync } from "node:fs";
import QRCode from "qrcode";
import jsQR from "jsqr";
import { pack, unpack } from "./occipital.mjs";

const ffmpegBin = async () => process.env.FFMPEG_PATH || (await import("ffmpeg-static").then((m) => m.default, () => null)) || "ffmpeg";

async function ffmpeg(args) {
  const proc = spawn(await ffmpegBin(), ["-hide_banner", ...args], { stdio: ["pipe", "pipe", "pipe"] });
  let log = "";
  proc.stderr.on("data", (d) => (log += d));
  proc.stdin.on("error", () => {}); // a dying ffmpeg closes stdin; its exit code reports why
  const done = new Promise((resolve, reject) => {
    proc.on("error", reject);
    proc.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited with ${code}: ${log.trim().split("\n").slice(-3).join(" | ")}`))));
  });
  return { proc, done, log: () => log };
}

/** Write `documents` ({ id: text }) to `path` as an MP4 with one QR frame per chunk. `scale` is pixels per QR module
 *  and `crf` the H.264 quality (lower = larger, safer); both were picked so every frame decodes (see test-occipital).
 *  Returns { frames, bytes, textBytes }. */
export async function writeVideo(path, documents, { chunkSize = 500, overlap = 50, scale = 4, margin = 2, crf = 23, fps = 1 } = {}) {
  const payloads = await pack(documents, { chunkSize, overlap });
  if (!payloads.length) throw new Error("nothing to write: every document is empty");
  const codes = payloads.map((p) => QRCode.create([{ data: p, mode: "byte" }], { errorCorrectionLevel: "M" }).modules);
  let side = (Math.max(...codes.map((c) => c.size)) + 2 * margin) * scale;
  side += side & 1; // yuv420p needs even sides
  const { proc, done } = await ffmpeg(["-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", `${side}x${side}`, "-r", String(fps), "-i", "-",
    "-c:v", "libx264", "-tune", "stillimage", "-crf", String(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]);
  for (const code of codes) {
    const frame = Buffer.alloc(side * side, 255), at = (side - code.size * scale) >> 1;
    for (let r = 0; r < code.size; r++) {
      for (let c = 0; c < code.size; c++) {
        if (!code.get(r, c)) continue;
        for (let y = 0; y < scale; y++) frame.fill(0, (at + r * scale + y) * side + at + c * scale, (at + r * scale + y) * side + at + (c + 1) * scale);
      }
    }
    if (!proc.stdin.write(frame)) await new Promise((resolve) => proc.stdin.once("drain", resolve));
  }
  proc.stdin.end();
  await done;
  return { frames: payloads.length, bytes: statSync(path).size, textBytes: (Array.isArray(documents) ? documents.map(([, t]) => t) : Object.values(documents)).reduce((n, t) => n + Buffer.byteLength(t), 0) };
}

/** Read an Occipital MP4: decode every frame and rebuild the search. Returns unpack()'s { memory, lost } plus frames.
 *  ponytail: decodes every frame on read (about 50 frames/s); cache JSON.stringify(memory) next to big videos. */
export async function readVideo(path, options = {}) {
  const { proc, done, log } = await ffmpeg(["-i", path, "-f", "rawvideo", "-pix_fmt", "rgba", "-"]);
  proc.stdin.end();
  const payloads = [], waiting = [];
  let width, frame, filled = 0;
  const take = (data) => {
    for (let o = 0; o < data.length; ) {
      const n = Math.min(data.length - o, frame.length - filled);
      data.copy(frame, filled, o, o + n);
      (o += n), (filled += n);
      if (filled < frame.length) continue;
      const qr = jsQR(new Uint8ClampedArray(frame.buffer, frame.byteOffset, frame.length), width, frame.length / 4 / width, { inversionAttempts: "dontInvert" });
      payloads.push(qr ? Uint8Array.from(qr.binaryData) : null);
      filled = 0;
    }
  };
  for await (const data of proc.stdout) {
    if (!frame) {
      const size = log().match(/Stream #0:0.*Video: .*?, (\d+)x(\d+)[\s,[]/); // ffmpeg prints the streams before any output
      if (!size) { waiting.push(data); continue; }
      width = Number(size[1]);
      frame = Buffer.alloc(width * Number(size[2]) * 4);
      waiting.splice(0).forEach(take);
    }
    take(data);
  }
  await done;
  return { ...(await unpack(payloads, options)), frames: payloads.length };
}
