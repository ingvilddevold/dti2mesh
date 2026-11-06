"""
Maps a cleaned DTI tensor (native scanner-RAS; see check_dti.py /
clean_dti_data.py) onto a mesh built from the upright,
rotation-corrected segmentation (see reorient_upright.py).

The mesh lives in the corrected RAS frame; the DTI volume in the native
RAS frame. R (native->corrected rotation) and the pivot centroid, both
from reorient_upright.py, are needed twice: to map each mesh cell
centroid back to native RAS for sampling
(native_ras = centroid + R.T @ (corrected_ras - centroid)), and to
rotate the sampled tensor's own axes into the mesh frame
(D_corrected = R @ D_native @ R.T).
"""
import argparse

import nibabel as nib
import numpy as np
import pyvista as pv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mesh", help="tetrahedral mesh (.vtk/.vtu)")
    p.add_argument("native_seg", help="The ORIGINAL segmentation passed to reorient_upright.py "
                                       "(any orientation) -- canonicalized here the same way, "
                                       "to recompute the exact same rotation pivot/centroid")
    p.add_argument("rotation_npy", help="Rotation matrix saved by reorient_upright.py")
    p.add_argument("dti", help="Cleaned DTI tensor NIfTI, native scanner-RAS space")
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    mesh = pv.read(args.mesh)
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

    # Canonicalize exactly as reorient_upright.py does so the pivot matches
    native_img = nib.as_closest_canonical(nib.load(args.native_seg))
    native_data = native_img.get_fdata()
    idx = np.argwhere(native_data > 0)
    centroid_ras = nib.affines.apply_affine(native_img.affine, idx).mean(axis=0)
    print("Rotation pivot (native RAS centroid):", centroid_ras)

    R = np.load(args.rotation_npy)

    corrected_ras = mesh.cell_centers().points
    native_ras = centroid_ras + (corrected_ras - centroid_ras) @ R

    dti_img = nib.load(args.dti)
    dti_data = dti_img.get_fdata()
    M1, M2, M3 = dti_data.shape[:3]
    ras2vox = np.linalg.inv(dti_img.affine)

    ijk = nib.affines.apply_affine(ras2vox, native_ras)
    ijk = np.rint(ijk).astype(int)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, M1 - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, M2 - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, M3 - 1)

    D = dti_data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    nonzero_frac = (np.abs(D).sum(axis=1) > 0).mean()
    print(f"Mapped DTI tensor onto {D.shape[0]} mesh cells "
          f"({nonzero_frac:.1%} nonzero).")

    # Rotate each tensor from native-RAS axes into the mesh's axes.
    D3 = D.reshape(-1, 3, 3)
    D3 = R @ D3 @ R.T
    D = D3.reshape(-1, 9)

    mesh.cell_data["DTI"] = D
    mesh.save(args.out)
    print(f"Saved mesh + DTI tensor field to {args.out}")


if __name__ == "__main__":
    main()
