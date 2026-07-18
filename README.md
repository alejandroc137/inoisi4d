[![2607.XXXXX](https://img.shields.io/badge/arXiv-2607.XXXX-b31b1b.svg)](https://arxiv.org/abs/2607.XXXX) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/alejandroc137/inoisi4d/License.txt) [![GitHub repo stars](https://img.shields.io/github/stars/alejandroc137/inoisi4d?style=social)](https://github.com/alejandroc137/inoisi4d)

# inoisy+

`inoisy+` generates four-dimensional inhomogeneous, anisotropic Gaussian
random fields for time-dependent black-hole source models.  The C code solves
the SPDE on a source grid

```text
X^i = (t_s, x_s, y_s, z_s)
```

and the Python tools in `tools/` preview the same off-equatorial velocity and
correlation geometry, write HDF5 parameter files, and convert solver output into
visualization-friendly products.

The stochastic-field construction follows the SPDE/Matern-field approach used
by Lee & Gammie (2021), while the semi-analytic source and ray-tracing context
is connected to the AART paper by Cardenas-Avendano et al. (2023).  See the
references at the end of this file.

The use of inoisy+ in scientific publications must be properly acknowledged. Please cite:

_______
Cardenas-Avendano, A., Rubiera-Garcia, D. & Vincent, F. H. "An Off-Equatorial Gaussian Random Field Generator for Modeling Spatiotemporal Variability in Astrophysical Sources." [arXiv:2607.XXXX](https://arxiv.org/abs/) 
_______

We also request that inoisy+ modifications or extensions leading to a scientific publication be made public as free software. 

<center> <em>Feel free to use images and movies produced with this code (with attribution) for your next presentation! </em> </center>

_______
![GitHub last commit](https://img.shields.io/github/last-commit//alejandroc137/inoisi4d?style=flat)
_______

## Repository Layout

| Path | Purpose |
| --- | --- |
| `Makefile` | Build rules for the MPI/HDF5/HYPRE executable `inoisy4d`. |
| `src/main.c` | MPI driver, command-line parser, HYPRE grid/matrix/vector setup, solver selection, recursive SPDE solves, and HDF5 output. |
| `src/model_inoisy4d.c` | Four-dimensional grid spacing, time-periodic boundary condition, 33-point stencil, and matrix stencil insertion. |
| `src/param_inoisy4d.c` | Physical/model parameters, Kerr disk velocity, jet velocity, torus/jet correlation tensors, SPDE coefficients, source normalization, and HDF5 parameter I/O. |
| `src/hdf5_utils.c` | Thin HDF5 convenience layer. |
| `include/*.h` | C headers for the modules above. |
| `tools/inoisy4d_geometry_preview.py` | Python preview of the velocity field, torus/jet weights, triads, and local correlation tensor. |
| `tools/write_params_torusjet.py` | Writes an HDF5 `/params` file usable by both the C code and Python preview. |
| `tools/inoisy4d_to_visit_emissivity.py` | Converts C-code output into a standardized/lognormal emissivity HDF5+XDMF pair for VisIt. |
| `tools/inoisy4d_disk_trajectory_diagnostic.py` | Integrates diagnostic trajectories of the prescribed disk coordinate velocity. |

## Prerequisites

The C code requires a C compiler, MPI, HDF5 with MPI support, GSL, and HYPRE
compiled with support for four-dimensional SStruct grids.  On the cluster setup
used for these runs, loading the following modules is sufficient for the
compiler, MPI, HDF5, and GSL pieces:

```bash
module load compilers/gcc/12.3.0 libs/gsl/2.7.1 mpi/openmpi/4.1.6 libs/hdf5/1.14.6
```

HYPRE can be cloned from upstream:

```bash
git clone https://github.com/hypre-space/hypre.git
```

The important detail is that HYPRE must be configured with a maximum dimension
of four.  From `hypre/src`, run:

```bash
./configure --enable-bigint --enable-maxdim=4 AR="gcc-ar -rcs"
make install
```

Without `--enable-maxdim=4`, the four-dimensional `SStructGrid` construction
used by `inoisy+` will not be available.

The Python tools require:

```text
numpy
h5py
matplotlib
```

`inoisy4d_geometry_preview.py` currently enables LaTeX rendering in Matplotlib,
so it also expects a working LaTeX installation with `lmodern`.  If that is not
available on a machine used only for quick diagnostics, set `text.usetex` to
`False` in the script.

## Building the C Code

Edit `Makefile` so that the compiler wrapper and HYPRE installation match your
machine.  A typical setup is:

```make
CC        = h5pcc
HYPRE_DIR = /path/to/hypre/src/hypre
```

If your GSL module does not populate include/library paths automatically, also
set `GSL_DIR` or adjust `CINCLUDES` and `LDFLAGS` in the `Makefile`.

Then compile:

```bash
make inoisy4d
```

The executable is written as:

```text
./inoisy4d
```

## Example C Run

The output directory must already exist.

```bash
mpirun -n 16 ./inoisy4d \
  -nl 64 -nj 64 -ni 64 -nk 4 \
  -pgrid 1 1 1 16 \
  -solver 0 \
  -nrecur 2 \
  -timer \
  -output /path/where/you/want/the/result
```

This example uses 16 MPI ranks, all decomposed along the time direction.  The
local grid on each rank is `(ni,nj,nl,nk)=(64,64,64,4)`, which corresponds to
local `(z,y,x,t)` ordering in the C/HYPRE indexing.  The output dataset is stored
in HDF5 order `(t,x,y,z)`.

To use an HDF5 parameter file:

```bash
python3 tools/write_params_torusjet.py

mpirun -n 16 ./inoisy4d \
  -nl 64 -nj 64 -ni 64 -nk 4 \
  -pgrid 1 1 1 16 \
  -params tools/params_torusjet.h5 \
  -solver 0 \
  -nrecur 2 \
  -output /path/to/output
```

Using the same parameter file for the C run and the Python preview is the safest
way to keep the two implementations synchronized.

## C Runtime Options

These options are parsed by `src/main.c` and can be supplied after compilation.

| Option | Default | Meaning |
| --- | ---: | --- |
| `-n <n>` | `32` | Set all local grid lengths to `n` before per-axis overrides. |
| `-ni <n>` | `32` | Local number of cells in the HYPRE index-0 direction, corresponding to `z_s`. |
| `-nj <n>` | `32` | Local number of cells in the HYPRE index-1 direction, corresponding to `y_s`. |
| `-nl <n>` | `32` | Local number of cells in the HYPRE index-2 direction, corresponding to `x_s`. |
| `-nk <n>` | `32` | Local number of cells in the HYPRE index-3 direction, corresponding to `t_s`. |
| `-pgrid <pl> <pj> <pi> <pk>` | `1 1 1 num_procs` | MPI processor grid.  The order corresponds to `(x_s,y_s,z_s,t_s)`.  The product `pl*pj*pi*pk` must equal the MPI task count. |
| `-solver <ID>` | `0` | Linear solver. `0` = GMRES + BoomerAMG. `1` = PCG + BoomerAMG. |
| `-nrecur <n>` | `2` | Number of recursive second-order solves.  The default `2` implements the integer-order `nu=2`, `d=4` factorization. |
| `-params <file>`, `-p <file>` | none | Read model parameters from HDF5 datasets under `/params`.  Missing datasets are ignored. |
| `-source <file>`, `-s <file>` | none | Read `/data/data_raw` from an existing HDF5 file as the source field instead of drawing new white noise.  If `-nrecur` is not set, the code uses `nrecur=1` in this mode. |
| `-ps <file>`, `-sp <file>` | none | Use the same HDF5 file as both parameter file and source file. |
| `-output <dir>`, `-o <dir>` | `.` | Directory for the generated HDF5 output file. |
| `-dryrun` | off | Run without writing HDF5 output. |
| `-timer` | off | Print timing information from rank 0. |
| `-v <n_pre> <n_post>` | `1 1` | Parsed, but not currently connected to the selected HYPRE solvers. |
| `-dump` | off | Parsed, but currently not used to write intermediate recursive stages. |
| `-help`, `--help`, `help` | off | Print the built-in usage message. |

## C Model Parameters Read From HDF5

The C code reads the following datasets from `/params` when `-params` is given.
All are optional; missing datasets retain the compiled-in defaults from
`src/param_inoisy4d.c`.

| Dataset | Type | C default | Meaning |
| --- | --- | ---: | --- |
| `component` | int | `2` | Source component selector. `0` = disk/torus tensor only, `1` = jet tensor only, `2` = one composite torus+jet tensor. |
| `lam0` | float | `-1.0` | Disk correlation time.  If `lam0 <= 0`, the code uses the local orbital time `2*pi/|Omega|`. |
| `lam1` | float | `5.0` | Disk long correlation length along the torus-surface spiral direction. |
| `lam2` | float | `1.5` | Disk secondary torus-surface correlation length. |
| `lam3` | float | `0.6` | Disk normal/cross-section correlation length. |
| `jet_lam0` | float | `10.0` | Jet correlation time. |
| `jet_lam1` | float | `2.5` | Jet helical/poloidal correlation length. |
| `jet_lam2` | float | `2.5` | Jet polar/transverse correlation length. |
| `jet_lam3` | float | `2.5` | Jet third orthogonal correlation length. |
| `pitch` | float | `pi/20` | Disk spiral pitch angle on the torus surface. |
| `jet_chi0` | float | `pi/20` | Asymptotic jet helical angle. |
| `jet_chi_rho` | float | `2.0` | Cylindrical radius scale used to regularize the jet helical angle on the axis. |
| `torus_r0` | float | `12.0` | Torus center radius used by the disk weight and torus triad. |
| `torus_ar` | float | `8.0` | Torus radial width. |
| `torus_az` | float | `5.0` | Torus vertical width. |
| `jet_width` | float | `2.5` | Jet core width at `z=0`. |
| `jet_opening` | float | `0.30` | Linear opening coefficient in `R_j(z)=jet_width+jet_opening*|z|`. |
| `blend_floor` | float | `1.0e-4` | Small floor added to disk/jet weights in composite mode. |
| `normalize_weights` | int | `0` | If `0`, weights localize the raw field.  If `1`, weights interpolate the tensor orientation/scale without changing the total weight as strongly. |
| `spin` | float | `0.94` | Kerr spin `a`, clamped to `[-0.999999,0.999999]`. |
| `xi` | float | `1.0` | Angular-momentum normalization in the disk rotation law. |
| `delta` | float | `3.0` | Angular-momentum deformation parameter in the disk rotation law. |
| `sigma` | int | `1` | Orbit branch.  Non-negative values are mapped to `+1` prograde; negative values to `-1` retrograde. |
| `betar` | float | `0.8` | Radial interpolation parameter in the C compiled default.  The value is clamped to `[0,1]`. |
| `use_plunge` | int | `1` | Enables the optional spherical inner radial branch. |
| `enforce_timelike` | int | `1` | Clamps `Omega` into the local Kerr timelike interval when needed. |
| `jet_vpol` | float | `0.5` | Poloidal/radial speed in the simple jet advection model. |
| `jet_omega` | float | `0.0` | Rigid angular velocity in the simple jet advection model. |

### Synchronization Note

The current implementation has three places where defaults appear:
`src/param_inoisy4d.c`, `tools/inoisy4d_geometry_preview.py`, and
`tools/write_params_torusjet.py`.  Most values agree, but the current radial
branch defaults differ:

| Parameter | C compiled default | `geometry_preview.py` default | `write_params_torusjet.py` default |
| --- | ---: | ---: | ---: |
| `betar` | `1.0` | `0.7` | `0.7` |
| `use_plunge` | `1` | `0` | `1` |

For production runs or manuscript figures, prefer an explicit HDF5 parameter
file and pass it to both sides:

```bash
./inoisy4d -params tools/params_torusjet.h5 ...
python3 tools/inoisy4d_geometry_preview.py --params tools/params_torusjet.h5 ...
```

## C Values That Currently Require Editing and Recompilation

The following choices are not runtime options in the current implementation.
Changing them requires editing the indicated source file and recompiling.

| Quantity | Current value | Location | Meaning |
| --- | ---: | --- | --- |
| `x0start`, `x0end` | `0`, `256` | `src/param_inoisy4d.c` | Source-grid time domain. |
| `x1start`, `x1end` | `-30`, `30` | `src/param_inoisy4d.c` | Source-grid `x_s` domain. |
| `x2start`, `x2end` | `-30`, `30` | `src/param_inoisy4d.c` | Source-grid `y_s` domain. |
| `x3start`, `x3end` | `-50`, `50` | `src/param_inoisy4d.c` | Source-grid `z_s` domain. |
| `model_name` | `inoisyPlus` | `src/param_inoisy4d.c` | Prefix used in generated output file names. |
| `ksq()` | `1.0` | `src/param_inoisy4d.c` | Zeroth-order part of the elliptic operator. |
| `Nnu4_sq` | `96*pi*pi` | `src/param_inoisy4d.c` | White-noise normalization for `nu=2`, `d=4`. |
| Time periodicity | periodic in `t_s` only | `src/model_inoisy4d.c` | Spatial directions use the local stencil/boundary treatment, not periodic wrapping. |
| Stencil size | `33` entries | `src/model_inoisy4d.c` | Four pure second derivatives plus six mixed-derivative pairs in 4D. |
| Linear-solver tolerances | `1.0e-4` | `src/main.c` | GMRES/PCG convergence tolerance. |
| GMRES settings | max iter `100`, Krylov dimension `30` | `src/main.c` | Settings for `-solver 0`. |
| PCG settings | max iter `50` | `src/main.c` | Settings for `-solver 1`. |
| BoomerAMG preconditioner | max iter `1`, tol `0` | `src/main.c` | Preconditioner used by both solver options. |
| RNG base seed | `1` | `src/main.c` | Each rank uses `1 + myid`; the help text mentions `GSL_RNG_SEED`, but the current source resets the seed explicitly. |
| Output filename template | fixed format string | `src/param_inoisy4d.c` | Encodes component, grid size, domain size, and several model parameters. |

## SPDE Discretization and Solver Details

The default model is the integer-order four-dimensional Matern/SPDE case with
`nu=2`.  In four dimensions this is implemented as repeated applications of a
local second-order elliptic solve; the default is:

```text
nrecur = 2
```

The local covariance tensor is assembled from disk and/or jet contributions.
In composite mode, the code constructs one tensor,

```text
Lambda = w_disk Lambda_disk + w_jet Lambda_jet,
```

and solves one scalar SPDE.  The disk and jet are therefore not two sequential
SPDE stages.

The C implementation uses HYPRE as follows:

| Layer | Current implementation |
| --- | --- |
| Grid/matrix interface | `HYPRE_SStructGrid`, `HYPRE_SStructGraph`, `HYPRE_SStructMatrix`, and `HYPRE_SStructVector` with dimension `4`. |
| Matrix object type | `HYPRE_PARCSR`, obtained from the SStruct objects for the solver phase. |
| Stencil | 33-point 4D stencil, including mixed derivative pairs from the full anisotropic tensor. |
| Default solver | `-solver 0`: `HYPRE_ParCSRGMRES` with BoomerAMG preconditioning. |
| Alternative solver | `-solver 1`: `HYPRE_ParCSRPCG` with BoomerAMG preconditioning.  This is mainly useful for symmetric or nearly symmetric tests. |
| Preconditioner | `HYPRE_BoomerAMG`, one AMG iteration per Krylov step. |

GMRES is the default because the spatially varying full tensor and mixed
derivatives can make the assembled operator nonsymmetric.

## C Output File

The output file name is generated automatically by rank 0 and broadcast to all
ranks.  The main datasets are:

| HDF5 path | Shape/order | Meaning |
| --- | --- | --- |
| `/data/data_raw` | `(Nt,Nx,Ny,Nz)` | Raw Gaussian random field after the recursive SPDE solve. |
| `/data/lc_raw` | `(Nt)` | Spatially integrated raw field using cell volume `dx1*dx2*dx3`. |
| `/params/*` | scalars | Grid/domain metadata, MPI layout, raw-field statistics, RNG seed field, and the model parameters listed above. |

The solver also writes:

```text
/params/avg_raw
/params/var_raw
/params/min_raw
/params/max_raw
```

which are used by the visualization script to standardize the raw field.

## Python Tool: Geometry Preview

`tools/inoisy4d_geometry_preview.py` mirrors the Kerr disk velocity, jet
velocity, torus/jet weights, local triads, and correlation tensor implemented in
`src/param_inoisy4d.c`.  It is intended for checking the model before spending
cluster time on a full HYPRE solve.

### Geometry Preview Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--params <file>` | none | Optional HDF5 file with `/params` datasets. |
| `--plane {xy,xz,yz}` | `xz` | Slice to plot. |
| `--vector <name>` | `principal` | Vector field to draw.  Choices: `principal`, `disk1`, `disk2`, `disk3`, `jet1`, `jet2`, `jet3`, `diskv`, `jetv`. |
| `--background <name>` | `weights` | Scalar background.  Choices: `weights`, `disk_weight`, `jet_weight`, `detlambda`, `omega`, `disk_speed`, `jet_speed`, `vector_norm`. |
| `--extent <float>` | `50.0` | Plot range `[-extent,extent]` in both displayed coordinates. |
| `--n <int>` | `81` | Number of sample points per displayed direction. |
| `--stride <int>` | `5` | Quiver subsampling stride. |
| `--fixed <float>` | `0.0` | Fixed coordinate not shown in the selected plane. |
| `--arrow-mode <mode>` | `auto` | `raw`, `unit3d`, `unit2d`, `none`, or `auto`.  `auto` uses raw arrows for velocities and projected unit arrows for basis vectors. |
| `--quiver-scale <float>` | none | Passed to Matplotlib `quiver`; smaller values make arrows longer. |
| `--out <file>` | auto | Output image name. |

### Geometry Preview Examples

Disk coordinate velocity in the equatorial plane:

```bash
python3 tools/inoisy4d_geometry_preview.py \
  --params tools/params_torusjet.h5 \
  --plane xy \
  --vector diskv \
  --background disk_speed \
  --arrow-mode raw \
  --out xy_disk_speed.png
```

Jet coordinate velocity in a meridional slice:

```bash
python3 tools/inoisy4d_geometry_preview.py \
  --params tools/params_torusjet.h5 \
  --plane xz \
  --vector jetv \
  --background jet_speed \
  --arrow-mode raw \
  --out xz_jet_speed.png
```

Principal spatial correlation direction:

```bash
python3 tools/inoisy4d_geometry_preview.py \
  --params tools/params_torusjet.h5 \
  --plane xz \
  --vector principal \
  --background detlambda \
  --out xz_principal_detlambda.png
```

## Python Tool: Parameter File Writer

`tools/write_params_torusjet.py` writes:

```text
tools/params_torusjet.h5
```

with a `/params` group containing the floating-point and integer datasets read
by the C code and the preview script.  Run it with:

```bash
python3 tools/write_params_torusjet.py
```

This script currently has no command-line options; edit the dictionaries
`params_f64` and `params_i32` in the script to change its defaults.

## Python Tool: VisIt Emissivity Export

`tools/inoisy4d_to_visit_emissivity.py` reads a C-code output file, standardizes
the raw Gaussian field,

```text
Fhat = (F - <F>) / std(F),
```

maps it to a positive lognormal torus+jet emissivity, and writes a new HDF5 file
plus an XDMF file.  Open the `.xmf` file in VisIt.

Basic use:

```bash
python3 tools/inoisy4d_to_visit_emissivity.py output.h5 --force
```

Subsample and write disk/jet components:

```bash
python3 tools/inoisy4d_to_visit_emissivity.py output.h5 \
  -o output_emissivity_visit \
  --time-stride 4 \
  --space-stride 2 \
  --sigma-d 0.2 \
  --sigma-j 0.2 \
  --write-components \
  --force
```

Useful options include:

| Option | Default | Meaning |
| --- | ---: | --- |
| `input` | required | C-code HDF5 output file. |
| `-o`, `--output-prefix` | `<input>_emissivity_visit` | Output prefix for `.h5` and `.xmf`. |
| `--dataset` | `/data/data_raw` | Input dataset to convert. |
| `--time-stride` | `1` | Export every Nth time slice. |
| `--space-stride` | `1` | Export every Nth spatial cell in each direction. |
| `--first`, `--last` | `0`, final index | First/last time index to export. |
| `--stats-mode` | `auto` | `auto`, `params`, `full`, or `exported` statistics for standardization. |
| `--mean`, `--std` | none | Explicit standardization override; use together. |
| `--sigma-d`, `--sigma-j` | `/params` or `0.2` | Disk/jet lognormal amplitudes. |
| `--jdisk0`, `--jjet0` | `/params` or `1.0` | Disk/jet envelope normalizations. |
| `--torus-r0`, `--torus-ar`, `--torus-az` | `/params` or `12,8,5` | Torus envelope parameters. |
| `--jet-width`, `--jet-opening` | `/params` or `2.5,0.30` | Jet envelope parameters. |
| `--jet-side {both,up,down}` | `both` | Two-sided or one-sided jet envelope. |
| `--write-components` | off | Also write `j_disk` and `j_jet`. |
| `--write-envelopes` | off | Also write deterministic envelopes. |
| `--write-raw` | off | Also write raw unstandardized field. |
| `--no-fhat` | off | Do not write `Fhat`. |
| `--float32` | off | Store visualization arrays as 32-bit floats. |
| `--compress` | off | Gzip-compress output datasets. |
| `--force` | off | Overwrite existing output files. |

## Python Tool: Disk Trajectory Diagnostic

`tools/inoisy4d_disk_trajectory_diagnostic.py` imports the equations from
`inoisy4d_geometry_preview.py` and integrates the prescribed coordinate
velocity `dx^i/dt = u^i/u^t`.  These are not geodesics; this is a diagnostic of
the current fluid ansatz.

Run:

```bash
python3 tools/inoisy4d_disk_trajectory_diagnostic.py \
  --params tools/params_torusjet.h5 \
  --outdir tools/trajectory_diagnostic
```

Options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--params` | `tools/params_torusjet.h5` | HDF5 parameter file. |
| `--outdir` | `tools/trajectory_diagnostic` | Directory for PNG diagnostics and `summary.md`. |
| `--dt` | `0.05` | Coordinate-time integration step. |
| `--max-abs` | `60.0` | Stop a trajectory if any coordinate exceeds this magnitude. |
| `--z-ell` | `0 5 15` | `z` slices for the angular-momentum/timelike-domain diagnostic. |

## Parameter Defaults Used By Python

The geometry preview uses the same parameter names as the C HDF5 interface.
Its default values are the dataclass values in `tools/inoisy4d_geometry_preview.py`.
The parameter writer uses the dictionaries in `tools/write_params_torusjet.py`.

| Parameter | Geometry preview default | Parameter-writer default |
| --- | ---: | ---: |
| `component` | `2` | `2` |
| `lam0` | `-1.0` | `-1.0` |
| `lam1` | `5.0` | `5.0` |
| `lam2` | `1.5` | `1.5` |
| `lam3` | `0.6` | `0.6` |
| `jet_lam0` | `10.0` | `10.0` |
| `jet_lam1` | `2.5` | `2.5` |
| `jet_lam2` | `2.5` | `2.5` |
| `jet_lam3` | `2.5` | `2.5` |
| `pitch` | `pi/20` | `pi/20` |
| `jet_chi0` | `pi/20` | `pi/20` |
| `jet_chi_rho` | `2.0` | `2.0` |
| `torus_r0` | `12.0` | `12.0` |
| `torus_ar` | `8.0` | `8.0` |
| `torus_az` | `5.0` | `5.0` |
| `jet_width` | `2.5` | `2.5` |
| `jet_opening` | `0.30` | `0.30` |
| `blend_floor` | `1.0e-4` | `1.0e-4` |
| `normalize_weights` | `0` | `0` |
| `spin` | `0.94` | `0.94` |
| `xi` | `1.0` | `1.0` |
| `delta` | `0.0` | `0.0` |
| `sigma` | `1` | `1` |
| `betar` | `0.8` | `0.8` |
| `use_plunge` | `0` | `0` |
| `enforce_timelike` | `1` | `1` |
| `jet_vpol` | `0.5` | `0.5` |
| `jet_omega` | `0.0` | `0.0` |

## References

Lee, D. and Gammie, C. F. 2021, "Disks as Inhomogeneous, Anisotropic Gaussian
Random Fields", Astrophysical Journal, 906, 39,
doi:10.3847/1538-4357/abc8f3.

Cardenas-Avendano, A., Lupsasca, A., and Zhu, H. 2023, "Adaptive analytical ray
tracing of black hole photon rings" (AART), Physical Review D, 107, 043030,
doi:10.1103/PhysRevD.107.043030.

Falgout, R. D., Jones, J. E., and Yang, U. M. 2005, "Pursuing Scalability for
hypre's Conceptual Interfaces", ACM Transactions on Mathematical Software, 31,
326-350.
