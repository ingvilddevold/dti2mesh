"""
Streamline fiber tractography (FACT-style) on the mesh's per-cell DTI
tensor field, rendered as a 3D scene of tract curves colored by local
direction (same red=L-R, green=A-P, blue=S-I DEC convention as
dec_map.py).


A streamline stops when it leaves the mesh, leaves the `white_matter`
cell mask, FA drops below `--fa-thresh`, or the direction turns sharper
than `--angle-thresh` (implausible for real fiber geometry).
"""
import argparse

import numpy as np
import pyvista as pv


def trace_half(mesh, dti, fa, wm, start_cell, seed, direction_sign,
                step_size, max_steps, fa_thresh, cos_angle_thresh):
    """Trace one half of a streamline from `seed`. `direction_sign`
    (+1/-1) picks which way along the seed cell's principal eigenvector
    to head initially; the two halves (+1 and -1) together form one
    full streamline through the seed."""
    points = [np.array(seed, dtype=float)]
    pos = np.array(seed, dtype=float)
    prev_dir = None
    cell_id = start_cell
    for _ in range(max_steps):
        if cell_id < 0 or not wm[cell_id] or fa[cell_id] < fa_thresh:
            break
        D = dti[cell_id].reshape(3, 3)
        _, eigvecs = np.linalg.eigh(D)
        v = eigvecs[:, -1]  # principal eigenvector (largest eigenvalue)
        if prev_dir is None:
            v = v * direction_sign
        else:
            if np.dot(v, prev_dir) < 0:
                v = -v  # pick the sign consistent with the incoming direction
            if np.dot(v, prev_dir) < cos_angle_thresh:
                break  # implausibly sharp turn
        pos = pos + step_size * v
        cell_id = mesh.find_containing_cell(pos)
        points.append(pos.copy())
        prev_dir = v
    return np.array(points)


def compute_streamlines(mesh, seeds, step_size, max_steps, fa_thresh,
                         angle_thresh_deg, min_points=5):
    dti = mesh.cell_data["DTI"]
    fa = mesh.cell_data["FA"]
    wm = mesh.cell_data["white_matter"].astype(bool)
    cos_thresh = np.cos(np.deg2rad(angle_thresh_deg))

    lines = []
    for seed in seeds:
        cell_id = mesh.find_containing_cell(seed)
        if cell_id < 0 or not wm[cell_id] or fa[cell_id] < fa_thresh:
            continue
        fwd = trace_half(mesh, dti, fa, wm, cell_id, seed, +1,
                          step_size, max_steps, fa_thresh, cos_thresh)
        bwd = trace_half(mesh, dti, fa, wm, cell_id, seed, -1,
                          step_size, max_steps, fa_thresh, cos_thresh)
        line = np.vstack([bwd[::-1], fwd[1:]])
        if len(line) >= min_points:
            lines.append(line)
    return lines


def lines_to_polydata(lines):
    """Build a pv.PolyData of all streamlines, with per-point RGB color
    from each point's local tangent direction (DEC convention)."""
    all_points = np.vstack(lines)
    connectivity = []
    offset = 0
    colors = np.zeros((len(all_points), 3), dtype=np.uint8)
    for line in lines:
        n = len(line)
        connectivity.append([n, *range(offset, offset + n)])

        tangents = np.gradient(line, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms > 0)
        colors[offset:offset + n] = (np.abs(tangents) * 255).astype(np.uint8)

        offset += n

    poly = pv.PolyData()
    poly.points = all_points
    poly.lines = np.hstack(connectivity)
    poly.point_data["DEC_RGB"] = colors
    return poly


def seed_points_in_white_matter(mesh, n_seeds, seed_rng):
    wm = mesh.cell_data["white_matter"].astype(bool)
    wm_idx = np.where(wm)[0]
    n_seeds = min(n_seeds, len(wm_idx))
    chosen = seed_rng.choice(wm_idx, size=n_seeds, replace=False)
    return mesh.cell_centers().points[chosen]


def render_view(poly, center, view, out_path, window_size=(1200, 1200)):
    """view: 'sagittal' (from -x), 'coronal' (from -y), 'axial' (from
    +z, looking down), or 'oblique' (3/4 perspective)."""
    d = 300
    cameras = {
        "sagittal": [(center[0] - d, center[1], center[2]), center, (0, 0, 1)],
        "coronal": [(center[0], center[1] - d, center[2]), center, (0, 0, 1)],
        "axial": [(center[0], center[1], center[2] + d), center, (0, 1, 0)],
    }
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.add_mesh(poly, scalars="DEC_RGB", rgb=True, line_width=1.5)
    plotter.background_color = "black"
    if view == "oblique":
        plotter.camera_position = "yz"
        plotter.camera.azimuth = 30
        plotter.camera.elevation = 15
    else:
        plotter.camera_position = cameras[view]
    plotter.screenshot(out_path)
    print(f"Saved {view} tractography render to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mesh", help="Mesh with 'DTI', 'FA', 'white_matter' cell arrays")
    p.add_argument("-o", "--out", required=True,
                    help="Output screenshot base path (.png); one file per view is "
                         "written as '{stem}_{view}{suffix}'")
    p.add_argument("--views", nargs="+", default=["sagittal", "coronal", "axial", "oblique"],
                    choices=["sagittal", "coronal", "axial", "oblique"])
    p.add_argument("--out-vtp", default=None,
                    help="Optional: also save the streamlines themselves as a .vtp "
                         "(for interactive viewing in ParaView etc.)")
    p.add_argument("--n-seeds", type=int, default=2000,
                    help="Seed points sampled from white-matter cell centroids")
    p.add_argument("--step-size", type=float, default=1.0, help="Integration step, mm")
    p.add_argument("--max-steps", type=int, default=300, help="Per half-streamline")
    p.add_argument("--fa-thresh", type=float, default=0.15,
                    help="Stop tracking when FA drops below this")
    p.add_argument("--angle-thresh", type=float, default=60.0,
                    help="Stop tracking on turns sharper than this (degrees)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for seed-point sampling")
    args = p.parse_args()

    mesh = pv.read(args.mesh)
    seeds = seed_points_in_white_matter(mesh, args.n_seeds, np.random.default_rng(args.seed))

    lines = compute_streamlines(
        mesh, seeds, args.step_size, args.max_steps, args.fa_thresh, args.angle_thresh,
    )
    print(f"Traced {len(lines)}/{len(seeds)} streamlines "
          f"({sum(len(l) for l in lines)} total points).")

    poly = lines_to_polydata(lines)
    if args.out_vtp:
        poly.save(args.out_vtp)
        print(f"Saved streamlines to {args.out_vtp}")

    from pathlib import Path
    out = Path(args.out)
    for view in args.views:
        out_path = out.with_name(f"{out.stem}_{view}{out.suffix}")
        render_view(poly, mesh.center, view, str(out_path))


if __name__ == "__main__":
    main()
