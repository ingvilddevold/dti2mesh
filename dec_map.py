"""
Render a directionally-encoded color (DEC) map of the fiber orientation
field on an axis-aligned slice through the mesh: the standard DTI
convention of RGB = |fiber axis| (red=left-right, green=anterior-posterior,
blue=superior-inferior), brightness scaled by FA.
"""
import argparse

import numpy as np
import pyvista as pv


def compute_dec_point_data(mesh, fa_thresh=0.15, bright_max=0.45, background_gray=55):
    D_pt = mesh.cell_data_to_point_data().point_data["DTI"].reshape(-1, 3, 3)

    eigvals, eigvecs = np.linalg.eigh(D_pt)
    l1, l2, l3 = eigvals[:, 2], eigvals[:, 1], eigvals[:, 0]
    fiber = eigvecs[:, :, 2]
    num = (l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2
    den = l1 ** 2 + l2 ** 2 + l3 ** 2
    FA = np.sqrt(0.5 * num / np.maximum(den, 1e-30))

    mask = FA > fa_thresh
    dec = np.abs(fiber) * np.clip((FA - fa_thresh) / (bright_max - fa_thresh), 0, 1)[:, None]
    rgb = np.full((len(FA), 3), background_gray, dtype=np.uint8)
    rgb[mask] = (np.clip(dec[mask], 0, 1) * 255).astype(np.uint8)

    mesh_pt = mesh.cell_data_to_point_data()
    mesh_pt.point_data["DEC_RGB"] = rgb
    mesh_pt.point_data["FA"] = FA
    return mesh_pt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mesh", help="Mesh with a 9-component 'DTI' cell array")
    p.add_argument("-o", "--out", required=True, help="Output screenshot (.png)")
    p.add_argument("--plane", choices=["axial", "coronal", "sagittal"], default="coronal")
    p.add_argument("--offset", type=float, default=10.0,
                    help="Offset (mm) from the mesh center along the slice normal")
    p.add_argument("--fa-thresh", type=float, default=0.15)
    p.add_argument("--bright-max", type=float, default=0.45)
    args = p.parse_args()

    mesh = pv.read(args.mesh)
    mesh_pt = compute_dec_point_data(mesh, args.fa_thresh, args.bright_max)

    normals = {"sagittal": (1, 0, 0), "coronal": (0, 1, 0), "axial": (0, 0, 1)}
    normal = normals[args.plane]
    origin = list(mesh.center)
    axis = normal.index(1)
    origin[axis] = args.offset if axis != 2 else args.offset  # offset along the slice normal

    sl = mesh_pt.slice(normal=normal, origin=origin)

    cam = {
        "coronal": [(mesh.center[0], -300, mesh.center[2]), (mesh.center[0], origin[1], mesh.center[2]), (0, 0, 1)],
        "axial": [(mesh.center[0], mesh.center[1], -300), (mesh.center[0], mesh.center[1], origin[2]), (0, 1, 0)],
        "sagittal": [(-300, mesh.center[1], mesh.center[2]), (origin[0], mesh.center[1], mesh.center[2]), (0, 0, 1)],
    }[args.plane]

    plotter = pv.Plotter(off_screen=True, window_size=(900, 900))
    plotter.add_mesh(sl, scalars="DEC_RGB", rgb=True, show_scalar_bar=False)
    plotter.camera_position = cam
    plotter.background_color = "black"
    plotter.screenshot(args.out)
    print(f"Saved {args.plane} DEC map to {args.out}")


if __name__ == "__main__":
    main()
