"""
Reorient a segmentation to standard RAS axis order and correct for head
tilt in the scanner, producing an "upright" volume suitable for meshing.

Two corrections are applied:

1. Axis reordering to ('R','A','S') via nibabel.as_closest_canonical.

2. A rigid rotation correcting for real head tilt in the scanner, derived
   from recon-all's talairach.xfm: the *pure rotation* component (from a
   polar decomposition of its 3x3 linear part, discarding the scale/shear
   that maps this subject's head size to the Talairach template) gives the
   native-space direction that talairach.xfm treats as "true superior".

Usage:
    python reorient_upright.py sub.nii.gz talairach.xfm --out sub_upright.nii.gz
"""

import argparse

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from scipy.linalg import polar


def parse_talairach_xfm(path):
    """Parse the 3x4 Linear_Transform matrix out of an MNI .xfm file."""
    text = open(path).read()
    start = text.index("Linear_Transform")
    nums = [
        float(x)
        for x in text[start:].replace(";", " ").split()
        if x.replace(".", "", 1)
        .replace("-", "", 1)
        .replace("e", "", 1)
        .replace("+", "", 1)
        .isdigit()
        or _is_float(x)
    ]
    return np.array(nums[:12]).reshape(3, 4)


def _is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def tilt_rotation_from_talairach(xfm_path):
    A = parse_talairach_xfm(xfm_path)[:, :3]
    R, _P = polar(A)
    return R


def reorient_upright(seg_path, out_path, talairach_xfm=None, out_shape=(320, 320, 320)):
    img = nib.load(seg_path)
    canon = nib.as_closest_canonical(img)
    data = canon.get_fdata()
    in_affine = canon.affine

    idx = np.argwhere(data > 0)
    centroid_ras = nib.affines.apply_affine(in_affine, idx).mean(axis=0)

    if talairach_xfm is not None:
        R = tilt_rotation_from_talairach(talairach_xfm)
    else:
        R = np.eye(3)

    out_affine = np.eye(4)
    out_affine[:3, 3] = centroid_ras - np.array(out_shape) / 2.0

    def homog(R3, t=None):
        M = np.eye(4)
        M[:3, :3] = R3
        if t is not None:
            M[:3, 3] = t
        return M

    rot_about_centroid = homog(R.T)
    rot_about_centroid[:3, 3] = centroid_ras - R.T @ centroid_ras

    in_affine_inv = np.linalg.inv(in_affine)
    M = in_affine_inv @ rot_about_centroid @ out_affine

    out_data = ndi.affine_transform(
        data,
        M[:3, :3],
        offset=M[:3, 3],
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0.0,
    ).astype(data.dtype)

    print(
        f"nonzero voxels: in={int((data > 0).sum())}, out={int((out_data > 0).sum())}"
    )

    out_img = nib.Nifti1Image(out_data, out_affine)
    nib.save(out_img, out_path)
    print(f"Saved upright segmentation to {out_path}")
    return R, centroid_ras


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("seg", help="Input segmentation (.nii.gz or .mgz), any orientation")
    p.add_argument(
        "--talairach-xfm",
        default=None,
        help="recon-all's mri/transforms/talairach.xfm, for tilt correction",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--out-shape", type=int, nargs=3, default=(320, 320, 320))
    p.add_argument(
        "--save-rotation",
        default=None,
        help="Optional .npy path to save the rotation matrix "
        "(needed later to map DTI/other native-space data "
        "onto the resulting mesh)",
    )
    args = p.parse_args()

    R, centroid = reorient_upright(
        args.seg, args.out, args.talairach_xfm, tuple(args.out_shape)
    )
    if args.save_rotation:
        np.save(args.save_rotation, R)
        print(f"Saved rotation matrix to {args.save_rotation}")
    print("centroid (native RAS, rotation pivot):", centroid)
