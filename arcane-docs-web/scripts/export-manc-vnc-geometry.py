#!/usr/bin/env python3
"""Export the Male CNS VNC neuropil mesh + traced neuron skeletons.

Shell: navis-flybrains `JRCFIB2022M_vnc.ply` (Janelia / Google Research complete
male Drosophila CNS). https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/

Neurons: published Male CNS v1.0 SWC skeletons from
`gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/`,
selected from the public body-annotations table. Coordinates are Male CNS
nanometres (SWC files store 8 nm voxels). They are remapped into the same
display space as the FlyWire brain so the cervical connective sits on the FAFB
cut, then clipped to the VNC volume so brain-only arbors of descending cells
do not appear in the wrong EM space.

Nothing in the arbor fill is procedural.
"""
from __future__ import annotations

import json
import urllib.request
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "vnc-meshes"
SKEL_CACHE = ROOT / ".cache" / "vnc-skeletons"
OUT_BIN = ROOT / "public" / "data" / "manc-vnc-geometry.bin"
OUT_META = ROOT / "src" / "data" / "manc-vnc-geometry-meta.json"

MESH_URL = "https://raw.githubusercontent.com/navis-org/navis-flybrains/main/flybrains/meshes/JRCFIB2022M_vnc.ply"
ANNOT_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
    "flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
SWC_BASE = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
    "skeletons-malecns/skeletons-swc/"
)

FLY_SCALE = 5.031442762457326e-06
NECK_Y = -0.40
VNC_REGION = 5
TUBE_SIDES = 5
VOXEL_NM = 8.0
PAD_NM = 40000.0
PRUNE_TWIGS_NM = 8000.0
TOLERANCE_NM = 2000.0
RADIUS_BOOST = 3.2
MIN_RADIUS = 0.0030
MAX_RADIUS = 0.0120
MAX_NEURONS = 96

VNC_SUPERCLASS = {
    "vnc_intrinsic": 40,
    "vnc_sensory": 22,
    "vnc_motor": 20,
    "ascending_neuron": 8,
    "descending_neuron": 6,
}
COLORS = {
    "vnc_motor": (0.25, 0.82, 0.88),
    "vnc_sensory": (0.32, 0.58, 0.98),
    "vnc_intrinsic": (0.78, 0.40, 0.95),
    "vnc_efferent": (0.95, 0.72, 0.32),
    "ascending_neuron": (0.95, 0.48, 0.72),
    "descending_neuron": (0.95, 0.86, 0.55),
}
STATUS_RANK = {
    "Reviewed": 0,
    "Traced": 1,
    "Roughly traced": 2,
    "Prelim Roughly traced": 3,
}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "arcane-manc-vnc"})
    with urllib.request.urlopen(req, timeout=90) as res:
        return res.read()


def parse_ply(data: bytes):
    header_end = data.find(b"end_header\n")
    if header_end < 0:
        raise ValueError("PLY header")
    header = data[:header_end].decode("ascii", "replace")
    body = data[header_end + len(b"end_header\n") :]
    n_vert = n_face = 0
    face_index = "i"
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n_vert = int(line.split()[-1])
        elif line.startswith("element face"):
            n_face = int(line.split()[-1])
        elif "vertex_indices" in line and "uint" in line:
            face_index = "I"
    verts = np.frombuffer(body, dtype="<f4", count=n_vert * 3).reshape(-1, 3).astype(np.float64)
    faces = []
    off = n_vert * 12
    idx_dtype = np.dtype("<u4") if face_index == "I" else np.dtype("<i4")
    for _ in range(n_face):
        count = body[off]
        off += 1
        idx = np.frombuffer(body, dtype=idx_dtype, count=count, offset=off)
        off += count * 4
        faces.append(idx.astype(np.int32))
    return verts, faces


def triangulate(faces):
    out = []
    for face in faces:
        for i in range(1, len(face) - 1):
            out.extend((int(face[0]), int(face[i]), int(face[i + 1])))
    return np.asarray(out, dtype=np.int32).reshape(-1, 3)


def vertex_normals(verts, tris):
    normals = np.zeros_like(verts)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face_n = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(normals, tris[:, k], face_n)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def display_origin(verts_nm: np.ndarray):
    cx = 0.5 * (verts_nm[:, 0].min() + verts_nm[:, 0].max())
    cy = 0.5 * (verts_nm[:, 1].min() + verts_nm[:, 1].max())
    zmin = float(verts_nm[:, 2].min())
    return cx, cy, zmin


def to_display(verts_nm: np.ndarray, cx: float, cy: float, zmin: float) -> np.ndarray:
    disp = np.empty_like(verts_nm)
    disp[:, 0] = (verts_nm[:, 0] - cx) * FLY_SCALE
    disp[:, 1] = NECK_Y - (verts_nm[:, 2] - zmin) * FLY_SCALE
    disp[:, 2] = (verts_nm[:, 1] - cy) * FLY_SCALE
    return disp


def parse_swc(text: str):
    ids, parents, xyz, radii = [], [], [], []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        ids.append(int(float(parts[0])))
        xyz.append((float(parts[2]), float(parts[3]), float(parts[4])))
        radii.append(float(parts[5]))
        parents.append(int(float(parts[6])))
    if not ids:
        raise ValueError("empty SWC")
    id_to_i = {n: i for i, n in enumerate(ids)}
    points = np.asarray(xyz, dtype=np.float64) * VOXEL_NM
    rad_nm = np.asarray(radii, dtype=np.float64) * VOXEL_NM
    links = []
    for i, parent in enumerate(parents):
        if parent < 0 or parent not in id_to_i:
            continue
        links.append((i, id_to_i[parent]))
    return points, np.asarray(links, dtype=np.int32).reshape(-1, 2), rad_nm


def clip_skeleton(points, links, radii, mn, mx):
    keep = np.all((points >= mn) & (points <= mx), axis=1)
    if int(keep.sum()) < 8:
        return None
    old_to_new = np.full(len(points), -1, dtype=np.int32)
    old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    new_links = []
    for a, b in links:
        na, nb = old_to_new[int(a)], old_to_new[int(b)]
        if na >= 0 and nb >= 0:
            new_links.append((na, nb))
    if len(new_links) < 6:
        return None
    return points[keep], np.asarray(new_links, dtype=np.int32), radii[keep]


def largest_component(points, links, radii):
    adj = [[] for _ in range(len(points))]
    for a, b in links:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen = np.zeros(len(points), dtype=bool)
    best = []
    for start in range(len(points)):
        if seen[start] or not adj[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp = [start]
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
                    comp.append(nxt)
        if len(comp) > len(best):
            best = comp
    if len(best) < 8:
        return None
    keep = np.zeros(len(points), dtype=bool)
    keep[best] = True
    old_to_new = np.full(len(points), -1, dtype=np.int32)
    old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    new_links = [(old_to_new[a], old_to_new[b]) for a, b in links if keep[a] and keep[b]]
    return points[keep], np.asarray(new_links, dtype=np.int32), radii[keep]


def simplify_path(path, points, tolerance):
    if len(path) <= 2:
        return path
    selected = {0, len(path) - 1}
    pending = [(0, len(path) - 1)]
    p = points[path]
    while pending:
        start, end = pending.pop()
        if end - start < 2:
            continue
        vector = p[end] - p[start]
        norm = float(vector @ vector)
        interior = p[start + 1 : end]
        fraction = np.clip((interior - p[start]) @ vector / max(norm, 1e-12), 0, 1)
        distances = np.linalg.norm(interior - (p[start] + fraction[:, None] * vector), axis=1)
        index = int(np.argmax(distances))
        if distances[index] > tolerance:
            split = start + 1 + index
            selected.add(split)
            pending.extend([(start, split), (split, end)])
    return [path[i] for i in sorted(selected)]


def prune_twigs(points, links, min_length, rounds=4):
    links = [(int(a), int(b)) for a, b in links]
    for _ in range(rounds):
        adjacent = {}
        for a, b in links:
            adjacent.setdefault(a, []).append(b)
            adjacent.setdefault(b, []).append(a)
        leaves = [n for n, nbrs in adjacent.items() if len(nbrs) == 1]
        drop = set()
        for leaf in leaves:
            path = [leaf, adjacent[leaf][0]]
            length = float(np.linalg.norm(points[path[0]] - points[path[1]]))
            while len(adjacent.get(path[-1], [])) == 2:
                nxt = next(n for n in adjacent[path[-1]] if n != path[-2])
                length += float(np.linalg.norm(points[path[-1]] - points[nxt]))
                path.append(nxt)
            if length < min_length:
                for u, v in zip(path[:-1], path[1:]):
                    drop.add((u, v) if u < v else (v, u))
        if not drop:
            break
        links = [(a, b) for a, b in links if (min(a, b), max(a, b)) not in drop]
    return np.asarray(links, dtype=np.int32).reshape(-1, 2)


def simplify_skeleton(points, links, tolerance):
    adjacent = [[] for _ in range(len(points))]
    for a, b in links:
        a, b = int(a), int(b)
        adjacent[a].append(b)
        adjacent[b].append(a)
    seen = set()
    result = []
    starts = [i for i, nbrs in enumerate(adjacent) if len(nbrs) != 2]
    starts += [i for i, nbrs in enumerate(adjacent) if len(nbrs) == 2]
    for start in starts:
        for neighbor in adjacent[start]:
            edge = tuple(sorted((start, neighbor)))
            if edge in seen:
                continue
            seen.add(edge)
            path = [start, neighbor]
            while len(adjacent[path[-1]]) == 2:
                nxt = next(n for n in adjacent[path[-1]] if n != path[-2])
                edge = tuple(sorted((path[-1], nxt)))
                if edge in seen:
                    break
                seen.add(edge)
                path.append(nxt)
            simple = simplify_path(path, points, tolerance)
            if len(simple) >= 2:
                result.append([int(i) for i in simple])
    return result


def _frame(axis: np.ndarray):
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    n = np.linalg.norm(u)
    if n < 1e-9:
        u = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(u)
    u = u / max(n, 1e-12)
    v = np.cross(axis, u)
    return u, v / max(np.linalg.norm(v), 1e-12)


def parallel_transport(points):
    n = len(points)
    tangents = np.empty_like(points)
    tangents[:-1] = points[1:] - points[:-1]
    tangents[-1] = tangents[-2]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(lengths, 1e-12)
    us = np.empty_like(points)
    vs = np.empty_like(points)
    u, _ = _frame(tangents[0])
    for i in range(n):
        t = tangents[i]
        u = u - t * float(np.dot(u, t))
        norm = float(np.linalg.norm(u))
        if norm < 1e-6:
            u, _ = _frame(t)
        else:
            u = u / norm
        us[i] = u
        vs[i] = np.cross(t, u)
    return us, vs


def tube_mesh(disp, radii_disp, paths):
    ring = np.linspace(0.0, 2.0 * np.pi, TUBE_SIDES, endpoint=False)
    cos_t = np.cos(ring)[None, :, None]
    sin_t = np.sin(ring)[None, :, None]
    pos_parts, nrm_parts = [], []
    for path in paths:
        pts = disp[path]
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9
        idx = np.asarray(path)[keep]
        pts = disp[idx]
        if len(pts) < 2:
            continue
        us, vs = parallel_transport(pts)
        radial = cos_t * us[:, None, :] + sin_t * vs[:, None, :]
        radii = np.clip(radii_disp[idx] * RADIUS_BOOST, MIN_RADIUS, MAX_RADIUS)
        rings = pts[:, None, :] + radial * radii[:, None, None]
        a, b = rings[:-1], rings[1:]
        aj, bj = np.roll(a, -1, axis=1), np.roll(b, -1, axis=1)
        na, nb = radial[:-1], radial[1:]
        naj, nbj = np.roll(na, -1, axis=1), np.roll(nb, -1, axis=1)
        pos_parts.append(np.stack([a, b, bj, a, bj, aj], axis=2).reshape(-1, 3))
        nrm_parts.append(np.stack([na, nb, nbj, na, nbj, naj], axis=2).reshape(-1, 3))
    if not pos_parts:
        return None
    normals = np.concatenate(nrm_parts)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return np.concatenate(pos_parts).astype(np.float32), normals.astype(np.float32)


def select_neurons(annot_path: Path, mesh_mn, mesh_mx):
    table = feather.read_table(annot_path)
    body = table.column("bodyId").to_pylist()
    sc = table.column("superclass").to_pylist()
    types = table.column("type").to_pylist()
    status = table.column("status").to_pylist()
    status_label = table.column("statusLabel").to_pylist()
    side = table.column("somaSide").to_pylist()
    neu = table.column("somaNeuromere").to_pylist()
    soma = table.column("somaLocation").to_pylist()

    buckets = defaultdict(list)
    for i, superclass in enumerate(sc):
        if superclass not in VNC_SUPERCLASS:
            continue
        rank = STATUS_RANK.get(status_label[i], STATUS_RANK.get(status[i], 9))
        if rank > 3:
            continue
        loc = soma[i]
        # Sensory somas often sit in the periphery; descending somas sit in the
        # brain. Keep both and clip the arbor to the VNC volume later.
        if superclass not in ("descending_neuron", "vnc_sensory"):
            if loc is None:
                continue
            p = np.asarray(loc, dtype=np.float64) * VOXEL_NM
            if np.any(p < mesh_mn - PAD_NM) or np.any(p > mesh_mx + PAD_NM):
                continue
        typ = types[i] or f"body{body[i]}"
        buckets[superclass].append(
            {
                "id": int(body[i]),
                "superclass": superclass,
                "type": str(typ),
                "side": side[i] or "U",
                "neuromere": neu[i] or "NA",
                "rank": rank,
            }
        )

    chosen = []
    used_types: dict[tuple, int] = {}
    for superclass, quota in VNC_SUPERCLASS.items():
        cands = sorted(
            buckets[superclass],
            key=lambda n: (n["rank"], n["type"], n["id"]),
        )
        by_key = defaultdict(list)
        for n in cands:
            by_key[(n["neuromere"], n["side"])].append(n)
        keys = list(by_key.keys())
        picked = 0
        while picked < quota and keys:
            progressed = False
            for key in list(keys):
                while by_key[key]:
                    n = by_key[key].pop(0)
                    type_key = (superclass, n["type"])
                    if used_types.get(type_key, 0) >= 2:
                        continue
                    used_types[type_key] = used_types.get(type_key, 0) + 1
                    chosen.append(n)
                    picked += 1
                    progressed = True
                    break
                if not by_key[key]:
                    keys.remove(key)
                if picked >= quota:
                    break
            if not progressed:
                break

    # Always include Giant Fiber if present — its VNC axon is a landmark.
    if not any(n["type"] == "DNp01" for n in chosen):
        for n in buckets.get("descending_neuron", []):
            if n["type"] == "DNp01":
                chosen.append(n)
                break

    # De-dupe ids, cap.
    seen = set()
    unique = []
    for n in chosen:
        if n["id"] in seen:
            continue
        seen.add(n["id"])
        unique.append(n)
        if len(unique) >= MAX_NEURONS:
            break
    return unique


def load_swc(body_id: int) -> bytes:
    path = SKEL_CACHE / f"{body_id}.swc"
    if not path.exists():
        path.write_bytes(fetch(f"{SWC_BASE}{body_id}.swc"))
    return path.read_bytes()


def process_neuron(neuron, mn, mx, cx, cy, zmin):
    raw = load_swc(neuron["id"]).decode("ascii", "replace")
    points, links, radii = parse_swc(raw)
    clipped = clip_skeleton(points, links, radii, mn, mx)
    if clipped is None:
        return None
    points, links, radii = clipped
    component = largest_component(points, links, radii)
    if component is None:
        return None
    points, links, radii = component
    pruned = prune_twigs(points, links, PRUNE_TWIGS_NM)
    if len(pruned) < 6:
        return None
    paths = simplify_skeleton(points, pruned, TOLERANCE_NM)
    if not paths:
        return None
    disp = to_display(points, cx, cy, zmin)
    radii_disp = radii * FLY_SCALE
    swept = tube_mesh(disp, radii_disp, paths)
    if swept is None:
        return None
    pos, nrm = swept
    rgb = np.asarray(COLORS.get(neuron["superclass"], (0.7, 0.45, 0.9)), dtype=np.float32)
    col = np.repeat(rgb[None, :], len(pos), axis=0)
    return pos, nrm, col, neuron


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    SKEL_CACHE.mkdir(parents=True, exist_ok=True)
    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)

    mesh_path = CACHE / "JRCFIB2022M_vnc.ply"
    if not mesh_path.exists():
        print("Fetching Male CNS VNC mesh…", flush=True)
        mesh_path.write_bytes(fetch(MESH_URL))
    annot_path = CACHE / "body-annotations.feather"
    if not annot_path.exists() or annot_path.stat().st_size < 1000:
        print("Fetching Male CNS body annotations…", flush=True)
        annot_path.write_bytes(fetch(ANNOT_URL))

    verts_nm, faces = parse_ply(mesh_path.read_bytes())
    tris = triangulate(faces)
    cx, cy, zmin = display_origin(verts_nm)
    disp = to_display(verts_nm, cx, cy, zmin)
    nrm = vertex_normals(disp, tris)

    mesh_positions = disp[tris].reshape(-1, 3).astype(np.float32)
    mesh_normals = nrm[tris].reshape(-1, 3).astype(np.float32)
    mesh_regions = np.full(len(mesh_positions), VNC_REGION, dtype=np.uint8)

    mn = verts_nm.min(0) - PAD_NM
    mx = verts_nm.max(0) + PAD_NM
    neurons = select_neurons(annot_path, verts_nm.min(0), verts_nm.max(0))
    print(f"Selected {len(neurons)} Male CNS VNC neurons", flush=True)
    for sc, quota in VNC_SUPERCLASS.items():
        n = sum(1 for x in neurons if x["superclass"] == sc)
        print(f"  {sc}: {n}/{quota}", flush=True)

    pos_parts, nrm_parts, col_parts = [], [], []
    kept = []
    print("Fetching and sweeping traced skeletons…", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(process_neuron, n, mn, mx, cx, cy, zmin) for n in neurons]
        for fut in as_completed(futs):
            try:
                result = fut.result()
            except Exception as err:
                print(f"  skip: {err}", flush=True)
                continue
            if result is None:
                continue
            pos, normals, col, neuron = result
            pos_parts.append(pos)
            nrm_parts.append(normals)
            col_parts.append(col)
            kept.append(neuron)
            print(
                f"  {neuron['id']} {neuron['type']} {neuron['superclass']} "
                f"{len(pos) // 3:,} tris",
                flush=True,
            )

    if not pos_parts:
        raise SystemExit("No VNC skeletons survived clipping")

    arbor_pos = np.concatenate(pos_parts).astype(np.float32)
    arbor_nrm = np.concatenate(nrm_parts).astype(np.float32)
    arbor_col = np.concatenate(col_parts).astype(np.float32)
    arbor_reg = np.full(len(arbor_pos), VNC_REGION, dtype=np.uint8)

    mesh_pos_f = mesh_positions.reshape(-1)
    mesh_nrm_f = mesh_normals.reshape(-1)
    arbor_pos_f = arbor_pos.reshape(-1)
    arbor_nrm_f = arbor_nrm.reshape(-1)
    arbor_col_f = arbor_col.reshape(-1)

    chunks = [
        mesh_pos_f.tobytes(),
        mesh_nrm_f.tobytes(),
        mesh_regions.tobytes(),
        arbor_pos_f.tobytes(),
        arbor_nrm_f.tobytes(),
        arbor_col_f.tobytes(),
        arbor_reg.tobytes(),
    ]
    mesh_count = int(len(mesh_pos_f) // 3)
    arbor_count = int(len(arbor_pos_f) // 3)
    mesh_pos_off = 0
    mesh_nrm_off = mesh_pos_f.nbytes
    mesh_reg_off = mesh_nrm_off + mesh_nrm_f.nbytes
    arbor_pos_off = mesh_reg_off + mesh_regions.nbytes
    arbor_nrm_off = arbor_pos_off + arbor_pos_f.nbytes
    arbor_col_off = arbor_nrm_off + arbor_nrm_f.nbytes
    arbor_reg_off = arbor_col_off + arbor_col_f.nbytes
    byte_length = arbor_reg_off + arbor_reg.nbytes

    OUT_BIN.write_bytes(b"".join(chunks))
    mn_d, mx_d = disp.min(0), disp.max(0)
    meta = {
        "dataset": "male-cns_vnc",
        "format": "triangles",
        "meshSource": MESH_URL,
        "skeletonSource": SWC_BASE,
        "citation": [
            "Berg et al., Cell 2026 — Sexual dimorphism in the complete connectome of the Drosophila male CNS",
            "navis-flybrains JRCFIB2022M_vnc neuropil mesh",
        ],
        "bin": "/data/manc-vnc-geometry.bin",
        "byteLength": byte_length,
        "tubeSides": TUBE_SIDES,
        "pruneTwigsNm": PRUNE_TWIGS_NM,
        "simplifyToleranceNm": TOLERANCE_NM,
        "neuronCount": len(kept),
        "neurons": [
            {
                "id": n["id"],
                "type": n["type"],
                "superclass": n["superclass"],
                "side": n["side"],
                "neuromere": n["neuromere"],
            }
            for n in sorted(kept, key=lambda x: x["id"])
        ],
        "bounds": {"min": mn_d.tolist(), "max": mx_d.tolist()},
        "mesh": {
            "count": mesh_count,
            "posOffset": mesh_pos_off,
            "nrmOffset": mesh_nrm_off,
            "regOffset": mesh_reg_off,
        },
        "arbors": {
            "count": arbor_count,
            "posOffset": arbor_pos_off,
            "nrmOffset": arbor_nrm_off,
            "colOffset": arbor_col_off,
            "regOffset": arbor_reg_off,
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUT_BIN} ({OUT_BIN.stat().st_size:,} bytes), "
        f"{mesh_count // 3:,} mesh triangles, {arbor_count // 3:,} arbor triangles, "
        f"{len(kept)} traced neurons",
        flush=True,
    )


if __name__ == "__main__":
    main()
