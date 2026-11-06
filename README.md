# DTI fiber-field mesh pipeline

Map a diffusion tensor image (DTI) onto a tetrahedral brain mesh as a
per-cell fiber field: clean up non-physiological tensor voxels, sample
the tensor onto the mesh in the mesh's own frame, and derive fiber
direction, fractional anisotropy (FA) and mean diffusivity (MD).

Builds on Chapter 5 of Mardal, Rognes, Thompson & Valnes, *Mathematical
Modeling of the Human Brain: From Magnetic Resonance Images to Finite
Element Simulation* (Simula SpringerBriefs on Computing, Springer, 2022),
[doi:10.1007/978-3-030-95136-8](https://doi.org/10.1007/978-3-030-95136-8).

### What comes from the book

- `check_dti.py` and `clean_dti_data.py` are the book's code
  (`mri2fem/chp5`, Section 5.2.1), used under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with two
  additions (see step 1): the tensor components are rotated from the DTI
  voxel axes into scanner axes, and `dt_recon`'s DTI-to-T1 registration
  can be applied.
- `dti_to_mesh.py` uses the same approach as the book's
  `dti_data_to_mesh.py` (Section 5.2.2): each mesh cell takes the tensor
  of the voxel nearest its center. It is rewritten for VTK meshes in
  scanner coordinates, and adds the head-tilt correction (step 4).
- `reorient_upright.py`, `fiber_direction.py`, `dec_map.py` and
  `fiber_tractography.py` are new.

## Inputs

For each subject you need:

- A FreeSurfer `recon-all` reconstruction (uses `mri/aseg.mgz`,
  `mri/wmparc.mgz` and `mri/transforms/talairach.xfm`).
- A DTI tensor volume from FreeSurfer's `dt_recon` (`tensor.nii.gz`).
- A tetrahedral mesh built from the segmentation (step 3;
  meshing is not part of this repo).


## Example data

The commands below use the "ernie" subject from the book's
[mri2fem dataset](https://zenodo.org/records/4899120) (not included in
this repo), which already includes the FreeSurfer and `dt_recon`
output. For other subjects, change the two variables:

```bash
FS=mri2fem-dataset/freesurfer/ernie          # recon-all subject directory
DTI=mri2fem-dataset/freesurfer/ernie-dti     # dt_recon output directory
```

## Pipeline

1. **Clean the DTI tensor** (`check_dti.py`, `clean_dti_data.py`):
   checks that eigenvalues are positive and 0 < FA < 1 within a dilated
   brain mask, and replaces invalid voxels with values from nearby valid
   ones. The output is resampled into the segmentation's native
   (scanner-RAS) voxel space.

   ```bash
   mri_binarize --i $FS/mri/wmparc.mgz --gm --dilate 2 --o mask.mgz
   python clean_dti_data.py --dti $DTI/tensor.nii.gz \
       --mask mask.mgz --reg $DTI/register.dat --out tensor-clean.nii.gz
   ```

   Two corrections not in the book's code:

   - **Tensor axes.** `dt_recon` stores the tensor components along the
     DTI image's voxel axes (for the mri2fem data roughly left,
     posterior, superior, with a ~14 degree slice angle), not along
     scanner axes. They are rotated into scanner axes,
     `D_ras = M @ D_vox @ M.T`. Without this, fiber directions are
     mirrored (the vertical tilt of oblique fibers is flipped) and off
     by the slice angle; FA and MD are unaffected. Use
     `--tensor-frame ras` for tensors that are already in scanner axes.
   - **Registration** (`--reg`). The book's code assumes the DTI and T1
     scans line up in scanner coordinates. `--reg` applies `dt_recon`'s
     `bbregister` result instead, both to positions and to the tensor
     axes (`D_t1 = Q.T @ D_dti @ Q`, with `Q` its rotation). Check the
     registration first, e.g. in `freeview` with `$DTI/lowb.nii.gz`
     over `$FS/mri/orig.mgz`. For ernie the registration cost (first
     number in `$DTI/register.dat.mincost`) is 0.27, and the scans
     differ by 1.4 degrees and 3 mm.

2. **Reorient to upright** (`reorient_upright.py`): reorders axes to RAS
   and corrects head tilt in the scanner, using the pure-rotation part of
   the subject's `talairach.xfm`. Saves the rotation for step 4.

   ```bash
   python reorient_upright.py $FS/mri/aseg.mgz \
       --talairach-xfm $FS/mri/transforms/talairach.xfm \
       --out aseg_upright.nii.gz --save-rotation tilt_rotation.npy
   ```

3. **Generate the mesh** (not part of this repo): build a tetrahedral
   mesh from `aseg_upright.nii.gz`, saved as e.g. `mesh_marked.vtk`. The mesh must be in the
   upright RAS frame of `aseg_upright.nii.gz` and carry a per-cell
   `marker` array with the FreeSurfer aseg labels (used to identify
   white matter in step 5).

4. **Map the DTI tensor onto the mesh** (`dti_to_mesh.py`):

   ```bash
   python dti_to_mesh.py mesh_marked.vtk $FS/mri/aseg.mgz \
       tilt_rotation.npy tensor-clean.nii.gz -o mesh_with_dti.vtu
   ```

   Pass the *original* segmentation here (the same file given to
   `reorient_upright.py`); the script uses it to recompute the rotation
   pivot.

   For each mesh cell, the centroid is rotated back to native RAS to
   pick the DTI voxel (nearest neighbor), and the sampled tensor is then
   rotated into the mesh frame, `D_mesh = R @ D_native @ R.T`. Without
   that second rotation the tensor stays in the original scanner axes, and fiber
   directions are off by the full head-tilt angle (about 17 degrees for
   ernie). FA and MD are unaffected because they depend only on the
   eigenvalues.

5. **Fiber direction, FA and MD** (`fiber_direction.py`): the fiber
   direction is the principal eigenvector of the tensor. It is only
   meaningful where FA is reasonably high (white matter).

   ```bash
   python fiber_direction.py mesh_with_dti.vtu -o mesh_with_fibers.vtu
   ```

6. **Visualize** (`dec_map.py`): directionally encoded color map on an
   axis-aligned slice (red = L-R, green = A-P, blue = S-I, brightness
   scaled by FA).

   ```bash
   python dec_map.py mesh_with_fibers.vtu -o dec_coronal.png --plane coronal
   ```

7. **Fiber tractography** (`fiber_tractography.py`): FACT-style 3D
   streamlines seeded in white matter, colored with the same convention.
   The principal eigenvector is recomputed from the tensor at each step,
   with its sign chosen to match the incoming direction, because
   eigenvectors have no fixed sign.

   ```bash
   python fiber_tractography.py mesh_with_fibers.vtu -o tracts.png
   ```

## Output

The final mesh (`mesh_with_fibers.vtu`) keeps the input `marker` array
and adds these cell arrays:

- `DTI`: diffusion tensor, 9 components
- `fiber_direction`: principal eigenvector
- `FA`, `MD`
- `white_matter`: boolean mask, cerebral and cerebellar white matter

