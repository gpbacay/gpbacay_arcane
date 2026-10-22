// Size and speed of an Occipital video, and whether it survives lossy re-encoding. node bench-occipital.mjs [folder of .md]
// The numbers on the docs site's Occipital page come from: node bench-occipital.mjs ../docs
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { gzipSync } from "node:zlib";
import ffmpegStatic from "ffmpeg-static";
import { writeVideo, readVideo } from "./occipital-video.mjs";

const folder = process.argv[2] ?? "../docs";
const docs = Object.fromEntries(readdirSync(folder).filter((f) => f.endsWith(".md")).map((f) => [f, readFileSync(path.join(folder, f), "utf8")]));
const text = Object.values(docs).join("");
const dir = mkdtempSync(path.join(tmpdir(), "occipital-bench-"));
const file = path.join(dir, "kb.mp4"), again = path.join(dir, "again.mp4");
try {
  let t = performance.now();
  const w = await writeVideo(file, docs);
  const write = performance.now() - t;
  t = performance.now();
  const o = await readVideo(file);
  const open = performance.now() - t;
  execFileSync(process.env.FFMPEG_PATH || ffmpegStatic, ["-v", "error", "-y", "-i", file, "-c:v", "libx264", "-crf", "35", again]);
  const re = await readVideo(again);
  console.table({
    documents: Object.keys(docs).length,
    "text bytes": w.textBytes,
    "gzip bytes": gzipSync(text).length,
    frames: w.frames,
    "video bytes": w.bytes,
    "video / text": +(w.bytes / w.textBytes).toFixed(2),
    "write ms": Math.round(write),
    "read ms": Math.round(open),
    "frames lost": o.lost,
    "re-encoded (crf 35) bytes": statSync(again).size,
    "re-encoded frames lost": re.lost,
  });
} finally {
  rmSync(dir, { recursive: true, force: true });
}
