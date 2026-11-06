"""
Check a DTI tensor field (e.g. tensor.nii.gz from FreeSurfer's dt_recon) for
non-physiological voxels (non-positive eigenvalues, or FA outside (0, 1)),
resampled into the voxel space of a reference mask.

The returned tensor components are expressed along the mask's scanner
axes (RAS):

- dt_recon stores tensor components along the DTI image's own voxel
  axes (for the mri2fem data: roughly L, P, S with a ~14 degree in-plane
  slice angle), not scanner axes. With tensor_frame="voxel" (default)
  they are rotated into scanner axes, D_ras = M D_vox M^T, where M holds
  the voxel axes' unit direction vectors from the image affine.
- If reg_file (dt_recon's register.dat) is given, the DTI-to-T1
  registration is applied: positions are sampled through it, and the
  tensor axes are rotated by its rotation part Q, D_t1 = Q^T D_dti Q.
  Without it, the DTI and T1 scans are assumed to be aligned in scanner
  coordinates.

Adapted from mri2fem/chp5/check_dti.py in Mardal, Rognes, Thompson &
Valnes, "Mathematical Modeling of the Human Brain" (Simula SpringerBriefs
on Computing, Springer, 2022), Section 5.2.1,
https://doi.org/10.1007/978-3-030-95136-8, licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).
Changes: formatting, command-line help text, the voxel-to-scanner
rotation of the tensor components, and the optional registration
(load_registration and the tensor_frame/reg_file arguments).
"""

import argparse

import nibabel
import numpy
from nibabel.processing import resample_from_to
from scipy.linalg import polar

numpy.seterr(divide="ignore", invalid="ignore")


def _vox2ras_tkr(image):
    """FreeSurfer's tkregister vox2ras for an image's grid (as tkregister2)."""
    header = nibabel.freesurfer.mghformat.MGHHeader()
    header.set_data_shape(image.shape[:3])
    header.set_zooms(image.header.get_zooms()[:3])
    return header.get_vox2ras_tkr()


def load_registration(reg_file, dti_image, t1_image):
    """
    Read a tkregister-style register.dat (as written by dt_recon/bbregister,
    with the DTI as the movable volume) and return the 4x4 map from T1
    scanner RAS to DTI scanner RAS.
    """
    R = numpy.loadtxt(reg_file, skiprows=4, max_rows=4)
    return (
        dti_image.affine
        @ numpy.linalg.inv(_vox2ras_tkr(dti_image))
        @ R
        @ _vox2ras_tkr(t1_image)
        @ numpy.linalg.inv(t1_image.affine)
    )


def compute_FA(lmbdas):
    MD = (lmbdas[:, 0] + lmbdas[:, 1] + lmbdas[:, 2]) / 3.0
    FA2 = (
        (3.0 / 2.0)
        * (
            (lmbdas[:, 0] - MD) ** 2
            + (lmbdas[:, 1] - MD) ** 2
            + (lmbdas[:, 2] - MD) ** 2
        )
        / (lmbdas[:, 0] ** 2 + lmbdas[:, 1] ** 2 + lmbdas[:, 2] ** 2)
    )
    FA = numpy.sqrt(FA2)
    return FA


def check_dti_data(dti_file, mask_file, order=0, reg_file=None, tensor_frame="voxel"):
    # Load the DTI image data and mask:
    dti_image = nibabel.load(dti_file)
    dti_data = dti_image.get_fdata()

    mask_image = nibabel.load(mask_file)
    mask = mask_image.get_fdata().astype(bool)

    # Examine the differences in shape
    print("dti shape", dti_data.shape)
    print("mask shape", mask.shape)
    M1, M2, M3 = mask.shape

    # Map from T1 scanner RAS to DTI scanner RAS (identity unless registered)
    if reg_file is not None:
        S = load_registration(reg_file, dti_image, mask_image)
        print("Applying DTI-to-T1 registration from", reg_file)
    else:
        S = numpy.eye(4)

    # Create an empty image as a helper for mapping
    # from DTI voxel space to T1 voxel space:
    shape = numpy.zeros((M1, M2, M3, 9))
    vox2ras = mask_image.header.get_vox2ras()
    Nii = nibabel.nifti1.Nifti1Image
    helper = Nii(shape, S @ vox2ras)

    # Resample the DTI data in the T1 voxel space.
    image = resample_from_to(dti_image, helper, order=order)
    D = image.get_fdata()

    # Reshape D from M1 x M2 x M3 x 9 into N x 3 x 3:
    D = D.reshape(-1, 3, 3)

    # Express the tensor components along T1 scanner axes:
    # DTI voxel axes -> DTI scanner axes, then registration rotation.
    if tensor_frame == "voxel":
        A = dti_image.affine[:3, :3]
        M = polar(A / numpy.linalg.norm(A, axis=0))[0]
    elif tensor_frame == "ras":
        M = numpy.eye(3)
    else:
        raise ValueError(f"tensor_frame must be 'voxel' or 'ras', not {tensor_frame!r}")
    Q = polar(S[:3, :3])[0]
    T = Q.T @ M
    D = T @ D @ T.T

    # Compute eigenvalues and eigenvectors
    lmbdas, v = numpy.linalg.eigh(D)

    # Compute fractional anisotropy (FA)
    FA = compute_FA(lmbdas)

    # Define valid entries as those where all eigenvalues are
    # positive and FA is between 0 and 1
    positives = (lmbdas[:, 0] > 0) * (lmbdas[:, 1] > 0) * (lmbdas[:, 2] > 0)
    valid = positives * (FA < 1.0) * (FA > 0.0)
    valid = valid.reshape((M1, M2, M3))

    # Find all voxels with invalid tensors within the mask
    ii, jj, kk = numpy.where((~valid) * mask)
    print("Number of invalid tensor voxels within the mask ROI:", len(ii))

    # Reshape D from N x 3 x 3 to M1 x M2 x M3 x 9
    D = D.reshape((M1, M2, M3, 9))

    return valid, mask, D


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dti", required=True, help="DTI tensor NIfTI (tensor.nii.gz)")
    parser.add_argument("--mask", required=True, help="Brain mask (mask.mgz)")
    parser.add_argument("--reg", default=None,
                        help="dt_recon's register.dat (DTI-to-T1 registration); "
                             "if omitted, scanner coordinates are assumed aligned")
    parser.add_argument("--tensor-frame", choices=["voxel", "ras"], default="voxel",
                        help="Axes of the stored tensor components: the DTI image's "
                             "voxel axes (dt_recon) or scanner RAS")
    args = parser.parse_args()

    check_dti_data(args.dti, args.mask, reg_file=args.reg, tensor_frame=args.tensor_frame)
