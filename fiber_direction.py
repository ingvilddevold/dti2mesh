"""
Compute the principal fiber direction from a per-cell DTI tensor field:
the eigenvector of the largest eigenvalue of the (symmetric) diffusion
tensor D. Water diffuses fastest along myelinated axons and is restricted
across them, so this is the local fiber orientation.

Also computes fractional anisotropy (FA) and mean diffusivity (MD).
Direction is only meaningful where FA is reasonably high (white matter);
in gray matter/CSF diffusion is close to isotropic and the "principal"
eigenvector is essentially noise.
"""

import argparse

import numpy as np
import pyvista as pv

# FreeSurfer aseg labels for white matter
# (cerebral + cerebellar, both hemispheres)
WHITE_MATTER_LABELS = [2, 41, 7, 46]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "mesh",
        help="Mesh with a 9-component 'DTI' cell array and a 'marker' cell array",
    )
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    mesh = pv.read(args.mesh)
    D = mesh.cell_data["DTI"].reshape(-1, 3, 3)

    # eigh: D is symmetric -> real eigenvalues (ascending) and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(D)
    l1, l2, l3 = eigvals[:, 2], eigvals[:, 1], eigvals[:, 0]  # largest -> smallest
    fiber_direction = eigvecs[:, :, 2]  # principal eigenvector

    MD = (l1 + l2 + l3) / 3
    num = (l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2
    den = l1**2 + l2**2 + l3**2
    FA = np.sqrt(0.5 * num / np.maximum(den, 1e-30))

    white_matter = np.isin(mesh.cell_data["marker"], WHITE_MATTER_LABELS).astype(
        np.uint8
    )

    mesh.cell_data["fiber_direction"] = fiber_direction
    mesh.cell_data["FA"] = FA
    mesh.cell_data["MD"] = MD
    mesh.cell_data["white_matter"] = white_matter

    mesh.save(args.out)
    print(f"Saved mesh with fiber_direction/FA/MD/white_matter to {args.out}")
    print(f"FA range: {FA.min():.3f} - {FA.max():.3f}, mean {FA.mean():.3f}")
    print(f"White matter cells: {white_matter.sum()} / {len(white_matter)}")


if __name__ == "__main__":
    main()
