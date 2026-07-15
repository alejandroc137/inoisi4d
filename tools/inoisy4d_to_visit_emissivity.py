#!/usr/bin/env python3
"""
Convert an inoisy4d/inoisy+ HDF5 output file into a VisIt-friendly
HDF5+XDMF pair, after standardizing the raw Gaussian field and mapping it
to a positive lognormal emissivity field.

Input convention
----------------
The solver writes a 4D scalar field with shape

    /data/data_raw[t, x, y, z]

or another dataset supplied with --dataset.  This script computes

    Fhat = (F - <F>) / sqrt(<F^2> - <F>^2)

and then writes

    j_s = jbar_d exp(sigma_d Fhat - sigma_d^2/2)
        + jbar_j exp(sigma_j Fhat - sigma_j^2/2).

The deterministic envelopes used by default are a thick torus and a
collimated two-sided jet,

    jbar_d = jdisk0 exp[-1/2 ((rho-r0)^2/ar^2 + z^2/az^2)],
    jbar_j = jjet0  exp[-1/2 rho^2 / R_j(z)^2],
    R_j(z) = jet_width + jet_opening |z|.

The output HDF5 file is transposed for VisIt/XDMF.  It should be used for
visualization only, not as a -source file for the SPDE solver.

Examples
--------
    python3 inoisy4d_to_visit_emissivity.py output.h5 --force

    python3 inoisy4d_to_visit_emissivity.py output.h5 \
        -o output_emissivity_visit \
        --time-stride 4 --space-stride 2 \
        --sigma-d 0.2 --sigma-j 0.2 \
        --write-components --force

By default, sigma_d=sigma_j=0.2 unless /params/sigma_d and /params/sigma_j
are present in the input file or command-line values are supplied.

Open the produced .xmf file in VisIt.
"""

from __future__ import annotations

import argparse
import posixpath
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


TINY = 1.0e-300


def read_scalar(f: h5py.File, path: str, default=None):
    """Read an HDF5 scalar dataset if present, otherwise return default."""
    if path in f:
        return f[path][()]
    return default


def read_param_float(f: h5py.File, name: str, default: float) -> float:
    """Read /params/<name> as a float, with a default."""
    value = read_scalar(f, f"/params/{name}", default)
    return float(np.asarray(value))


def as_float(x, name: str) -> float:
    if x is None:
        raise RuntimeError(f"Missing required scalar {name!r} in input file")
    return float(np.asarray(x))


def selected_indices(n: int, first: int, last: int | None, stride: int) -> list[int]:
    last_eff = n - 1 if last is None else min(last, n - 1)
    first_eff = max(first, 0)
    if first_eff > last_eff:
        raise RuntimeError(f"No indices selected: first={first_eff}, last={last_eff}, n={n}")
    return list(range(first_eff, last_eff + 1, stride))


def streaming_mean_var(
    dset: h5py.Dataset,
    time_indices: Iterable[int],
    sx: int,
    sy: int,
    sz: int,
) -> tuple[float, float, int]:
    """Compute population mean and variance without loading the full 4D field."""
    count = 0
    total = 0.0
    total2 = 0.0

    for it in time_indices:
        arr = np.asarray(dset[it, ::sx, ::sy, ::sz], dtype=np.float64)
        count += arr.size
        total += float(arr.sum(dtype=np.float64))
        total2 += float(np.square(arr, dtype=np.float64).sum(dtype=np.float64))

    if count <= 0:
        raise RuntimeError("Cannot compute statistics from an empty selection")

    mean = total / count
    var = total2 / count - mean * mean
    var = max(var, TINY)
    return mean, var, count


def compute_stats(
    fin: h5py.File,
    dset: h5py.Dataset,
    mode: str,
    export_times: list[int],
    sx: int,
    sy: int,
    sz: int,
    mean_override: float | None,
    std_override: float | None,
) -> tuple[float, float, str]:
    """Return mean, variance, and a human-readable source description."""
    if mean_override is not None or std_override is not None:
        if mean_override is None or std_override is None:
            raise RuntimeError("Use --mean and --std together, or neither")
        if std_override <= 0.0:
            raise RuntimeError("--std must be positive")
        return float(mean_override), float(std_override) ** 2, "command-line"

    if mode == "params":
        mean = read_scalar(fin, "/params/avg_raw", None)
        var = read_scalar(fin, "/params/var_raw", None)
        if mean is None or var is None:
            raise RuntimeError("--stats-mode params requires /params/avg_raw and /params/var_raw")
        var_f = max(float(np.asarray(var)), TINY)
        return float(np.asarray(mean)), var_f, "/params/avg_raw,/params/var_raw"

    if mode == "auto":
        mean = read_scalar(fin, "/params/avg_raw", None)
        var = read_scalar(fin, "/params/var_raw", None)
        if mean is not None and var is not None:
            var_f = max(float(np.asarray(var)), TINY)
            return float(np.asarray(mean)), var_f, "/params/avg_raw,/params/var_raw"
        mode = "full"

    nt = int(dset.shape[0])
    if mode == "full":
        times = range(nt)
        mean, var, count = streaming_mean_var(dset, times, 1, 1, 1)
        return mean, var, f"full input dataset, N={count}"

    if mode == "exported":
        mean, var, count = streaming_mean_var(dset, export_times, sx, sy, sz)
        return mean, var, f"exported subset, N={count}"

    raise RuntimeError(f"Unknown stats mode {mode!r}")


def coordinate_arrays(
    nx0: int,
    ny0: int,
    nz0: int,
    sx: int,
    sy: int,
    sz: int,
    x1start: float,
    x2start: float,
    x3start: float,
    dx1: float,
    dx2: float,
    dx3: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-center coordinates for the exported spatial subset."""
    ix = np.arange(0, nx0, sx, dtype=np.float64)
    iy = np.arange(0, ny0, sy, dtype=np.float64)
    iz = np.arange(0, nz0, sz, dtype=np.float64)
    x = x1start + dx1 * (ix + 0.5)
    y = x2start + dx2 * (iy + 0.5)
    z = x3start + dx3 * (iz + 0.5)
    return x, y, z


def torus_jet_envelopes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    torus_r0: float,
    torus_ar: float,
    torus_az: float,
    jet_width: float,
    jet_opening: float,
    jdisk0: float,
    jjet0: float,
    jet_side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic disk and jet envelopes in solver order [x,y,z]."""
    torus_ar = max(abs(torus_ar), 1.0e-12)
    torus_az = max(abs(torus_az), 1.0e-12)
    jet_width = max(abs(jet_width), 1.0e-12)
    jet_opening = max(float(jet_opening), 0.0)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    rho = np.sqrt(X * X + Y * Y)

    u = (rho - torus_r0) / torus_ar
    v = Z / torus_az
    jbar_d = jdisk0 * np.exp(-0.5 * (u * u + v * v))

    width = jet_width + jet_opening * np.abs(Z)
    jbar_j = jjet0 * np.exp(-0.5 * (rho / width) ** 2)

    if jet_side == "up":
        jbar_j = np.where(Z >= 0.0, jbar_j, 0.0)
    elif jet_side == "down":
        jbar_j = np.where(Z <= 0.0, jbar_j, 0.0)
    elif jet_side != "both":
        raise RuntimeError(f"Unknown jet_side {jet_side!r}")

    return jbar_d, jbar_j


def write_xmf(
    xmf_path: Path,
    h5_name: str,
    variables: dict[str, list[str]],
    time_values: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> None:
    """Write an XDMF temporal collection with one or more scalar attributes."""
    ox, oy, oz = origin
    dx, dy, dz = spacing

    with xmf_path.open("w", encoding="utf-8") as xmf:
        xmf.write('<?xml version="1.0" ?>\n')
        xmf.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        xmf.write('<Xdmf Version="2.0">\n')
        xmf.write('  <Domain>\n')
        xmf.write('    <Grid Name="inoisy4d" GridType="Collection" CollectionType="Temporal">\n')

        for n, tval in enumerate(time_values):
            xmf.write(f'      <Grid Name="step_{n:06d}" GridType="Uniform">\n')
            xmf.write(f'        <Time Value="{tval:.17g}"/>\n')
            xmf.write(
                f'        <Topology TopologyType="3DCoRectMesh" '
                f'NumberOfElements="{nz + 1} {ny + 1} {nx + 1}"/>\n'
            )
            xmf.write('        <Geometry GeometryType="Origin_DxDyDz">\n')
            xmf.write('          <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">\n')
            xmf.write(f'            {ox:.17g} {oy:.17g} {oz:.17g}\n')
            xmf.write('          </DataItem>\n')
            xmf.write('          <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">\n')
            xmf.write(f'            {dx:.17g} {dy:.17g} {dz:.17g}\n')
            xmf.write('          </DataItem>\n')
            xmf.write('        </Geometry>\n')

            for var_name, dsets in variables.items():
                dset = dsets[n]
                xmf.write(f'        <Attribute Name="{var_name}" AttributeType="Scalar" Center="Cell">\n')
                xmf.write(
                    f'          <DataItem Dimensions="{nz} {ny} {nx}" '
                    f'NumberType="Float" Precision="8" Format="HDF">\n'
                )
                xmf.write(f'            {h5_name}:{dset}\n')
                xmf.write('          </DataItem>\n')
                xmf.write('        </Attribute>\n')

            xmf.write('      </Grid>\n')

        xmf.write('    </Grid>\n')
        xmf.write('  </Domain>\n')
        xmf.write('</Xdmf>\n')


def add_variable_dataset(
    group: h5py.Group,
    variables: dict[str, list[str]],
    var_name: str,
    step: int,
    arr_xyz: np.ndarray,
    compression,
    dtype,
) -> None:
    """Store one [x,y,z] array as [z,y,x] and register it for XDMF."""
    arr_zyx = np.asarray(arr_xyz, dtype=dtype).transpose(2, 1, 0)
    dset_name = f"{var_name}_{step:06d}"
    group.create_dataset(dset_name, data=arr_zyx, compression=compression)
    variables.setdefault(var_name, []).append(posixpath.join("/data", dset_name))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize an inoisy4d raw Gaussian field, map it to a lognormal "
            "torus+jet emissivity field, and write a VisIt-readable HDF5+XDMF pair."
        )
    )
    parser.add_argument("input", help="inoisy4d output .h5 file")
    parser.add_argument(
        "-o", "--output-prefix", default=None,
        help="output prefix; default is <input-stem>_emissivity_visit"
    )
    parser.add_argument(
        "--dataset", default="/data/data_raw",
        help="input raw GRF dataset path, default /data/data_raw"
    )
    parser.add_argument(
        "--time-stride", type=int, default=1,
        help="export every Nth time slice, default 1"
    )
    parser.add_argument(
        "--space-stride", type=int, default=1,
        help="export every Nth cell in each spatial direction, default 1"
    )
    parser.add_argument("--first", type=int, default=0, help="first time index to export")
    parser.add_argument("--last", type=int, default=None, help="last time index to export, inclusive")
    parser.add_argument(
        "--stats-mode", choices=("auto", "params", "full", "exported"), default="auto",
        help=(
            "statistics for Fhat: auto uses /params/avg_raw and /params/var_raw if present, "
            "otherwise full; params requires those datasets; full uses the whole raw dataset; "
            "exported uses only the exported subset"
        )
    )
    parser.add_argument("--mean", type=float, default=None, help="override raw-field mean")
    parser.add_argument("--std", type=float, default=None, help="override raw-field standard deviation")

    parser.add_argument("--sigma-d", type=float, default=None, help="disk lognormal amplitude")
    parser.add_argument("--sigma-j", type=float, default=None, help="jet lognormal amplitude")
    parser.add_argument("--jdisk0", type=float, default=None, help="disk envelope normalization")
    parser.add_argument("--jjet0", type=float, default=None, help="jet envelope normalization")
    parser.add_argument("--torus-r0", type=float, default=None, help="torus center radius")
    parser.add_argument("--torus-ar", type=float, default=None, help="torus radial thickness")
    parser.add_argument("--torus-az", type=float, default=None, help="torus vertical thickness")
    parser.add_argument("--jet-width", type=float, default=None, help="jet base width")
    parser.add_argument("--jet-opening", type=float, default=None, help="linear jet opening coefficient")
    parser.add_argument(
        "--jet-side", choices=("both", "up", "down"), default="both",
        help="use a two-sided jet or only z>=0/z<=0, default both"
    )

    parser.add_argument("--write-components", action="store_true", help="also write j_disk and j_jet")
    parser.add_argument("--write-envelopes", action="store_true", help="also write jbar_disk and jbar_jet")
    parser.add_argument("--write-raw", action="store_true", help="also write the raw unstandardized field")
    parser.add_argument("--no-fhat", action="store_true", help="do not write Fhat diagnostic variable")
    parser.add_argument("--float32", action="store_true", help="write visualization arrays as float32")
    parser.add_argument("--compress", action="store_true", help="gzip-compress output datasets")
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    args = parser.parse_args()

    if args.time_stride < 1 or args.space_stride < 1:
        parser.error("--time-stride and --space-stride must be positive integers")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    prefix = args.output_prefix
    if prefix is None:
        prefix = str(input_path.with_name(input_path.stem + "_emissivity_visit"))
    prefix_path = Path(prefix).resolve()
    out_h5 = prefix_path.with_suffix(".h5")
    out_xmf = prefix_path.with_suffix(".xmf")

    if not args.force:
        for path in (out_h5, out_xmf):
            if path.exists():
                raise FileExistsError(f"{path} exists; use --force to overwrite")
    if out_h5.exists():
        out_h5.unlink()
    if out_xmf.exists():
        out_xmf.unlink()

    with h5py.File(input_path, "r") as fin:
        if args.dataset not in fin:
            raise RuntimeError(f"Input file does not contain {args.dataset!r}")
        dset = fin[args.dataset]
        if len(dset.shape) != 4:
            raise RuntimeError(f"Expected a 4D dataset at {args.dataset}; got shape {dset.shape}")

        nt, nx0, ny0, nz0 = map(int, dset.shape)
        time_indices = selected_indices(nt, args.first, args.last, args.time_stride)
        sx = sy = sz = args.space_stride
        nx = len(range(0, nx0, sx))
        ny = len(range(0, ny0, sy))
        nz = len(range(0, nz0, sz))

        x0start = as_float(read_scalar(fin, "/params/x0start"), "/params/x0start")
        dx0 = as_float(read_scalar(fin, "/params/dx0"), "/params/dx0")
        x1start = as_float(read_scalar(fin, "/params/x1start"), "/params/x1start")
        x2start = as_float(read_scalar(fin, "/params/x2start"), "/params/x2start")
        x3start = as_float(read_scalar(fin, "/params/x3start"), "/params/x3start")
        dx1_native = as_float(read_scalar(fin, "/params/dx1"), "/params/dx1")
        dx2_native = as_float(read_scalar(fin, "/params/dx2"), "/params/dx2")
        dx3_native = as_float(read_scalar(fin, "/params/dx3"), "/params/dx3")

        mean, var, stats_source = compute_stats(
            fin, dset, args.stats_mode, time_indices, sx, sy, sz, args.mean, args.std
        )
        std = np.sqrt(var)

        # Read envelope/lognormal parameters from /params when present, otherwise defaults.
        # Avoid using /params/sigma here because that is the prograde/retrograde sign in the velocity model.
        sigma_d = args.sigma_d if args.sigma_d is not None else read_param_float(fin, "sigma_d", 0.2)
        sigma_j = args.sigma_j if args.sigma_j is not None else read_param_float(fin, "sigma_j", 0.2)
        jdisk0 = args.jdisk0 if args.jdisk0 is not None else read_param_float(fin, "jdisk0", 1.0)
        jjet0 = args.jjet0 if args.jjet0 is not None else read_param_float(fin, "jjet0", 1.0)

        torus_r0 = args.torus_r0 if args.torus_r0 is not None else read_param_float(fin, "torus_r0", 12.0)
        torus_ar = args.torus_ar if args.torus_ar is not None else read_param_float(fin, "torus_ar", 8.0)
        torus_az = args.torus_az if args.torus_az is not None else read_param_float(fin, "torus_az", 5.0)
        jet_width = args.jet_width if args.jet_width is not None else read_param_float(fin, "jet_width", 2.5)
        jet_opening = args.jet_opening if args.jet_opening is not None else read_param_float(fin, "jet_opening", 0.30)

        x, y, z = coordinate_arrays(
            nx0, ny0, nz0, sx, sy, sz,
            x1start, x2start, x3start,
            dx1_native, dx2_native, dx3_native,
        )
        jbar_d, jbar_j = torus_jet_envelopes(
            x, y, z,
            torus_r0=torus_r0,
            torus_ar=torus_ar,
            torus_az=torus_az,
            jet_width=jet_width,
            jet_opening=jet_opening,
            jdisk0=jdisk0,
            jjet0=jjet0,
            jet_side=args.jet_side,
        )

        time_values = np.array([x0start + dx0 * (it + 0.0) for it in time_indices], dtype=np.float64)
        compression = "gzip" if args.compress else None
        dtype = np.float32 if args.float32 else np.float64
        variables: dict[str, list[str]] = {}

        with h5py.File(out_h5, "w") as fout:
            gout = fout.create_group("data")
            if "/params" in fin:
                fin.copy("/params", fout)
            else:
                fout.create_group("params")

            for n, it in enumerate(time_indices):
                raw = np.asarray(dset[it, ::sx, ::sy, ::sz], dtype=np.float64)
                fhat = (raw - mean) / std
                jdisk = jbar_d * np.exp(sigma_d * fhat - 0.5 * sigma_d * sigma_d)
                jjet = jbar_j * np.exp(sigma_j * fhat - 0.5 * sigma_j * sigma_j)
                emissivity = jdisk + jjet

                add_variable_dataset(gout, variables, "emissivity", n, emissivity, compression, dtype)
                if not args.no_fhat:
                    add_variable_dataset(gout, variables, "Fhat", n, fhat, compression, dtype)
                if args.write_components:
                    add_variable_dataset(gout, variables, "j_disk", n, jdisk, compression, dtype)
                    add_variable_dataset(gout, variables, "j_jet", n, jjet, compression, dtype)
                if args.write_envelopes:
                    add_variable_dataset(gout, variables, "jbar_disk", n, jbar_d, compression, dtype)
                    add_variable_dataset(gout, variables, "jbar_jet", n, jbar_j, compression, dtype)
                if args.write_raw:
                    add_variable_dataset(gout, variables, "raw", n, raw, compression, dtype)

                print(f"wrote time step {n:06d} from input index {it} / {nt - 1}")

            gout.create_dataset("time", data=time_values)
            p = fout["params"]
            p.attrs["visit_source_file"] = str(input_path)
            p.attrs["visit_source_dataset"] = args.dataset
            p.attrs["visit_original_shape_Nt_Nx_Ny_Nz"] = np.array(dset.shape, dtype=np.int64)
            p.attrs["visit_export_shape_Nt_Nx_Ny_Nz"] = np.array([len(time_indices), nx, ny, nz], dtype=np.int64)
            p.attrs["visit_time_stride"] = args.time_stride
            p.attrs["visit_space_stride"] = args.space_stride
            p.attrs["standardization_mean"] = mean
            p.attrs["standardization_var"] = var
            p.attrs["standardization_std"] = std
            p.attrs["standardization_source"] = stats_source
            p.attrs["emissivity_formula"] = "jbar_d*exp(sigma_d*Fhat-sigma_d^2/2)+jbar_j*exp(sigma_j*Fhat-sigma_j^2/2)"
            p.attrs["sigma_d"] = sigma_d
            p.attrs["sigma_j"] = sigma_j
            p.attrs["jdisk0"] = jdisk0
            p.attrs["jjet0"] = jjet0
            p.attrs["torus_r0"] = torus_r0
            p.attrs["torus_ar"] = torus_ar
            p.attrs["torus_az"] = torus_az
            p.attrs["jet_width"] = jet_width
            p.attrs["jet_opening"] = jet_opening
            p.attrs["jet_side"] = args.jet_side

    write_xmf(
        out_xmf,
        out_h5.name,
        variables,
        time_values,
        nx=nx,
        ny=ny,
        nz=nz,
        origin=(x1start, x2start, x3start),
        spacing=(dx1_native * sx, dx2_native * sy, dx3_native * sz),
    )

    print("\nStandardization:")
    print(f"  mean = {mean:.17g}")
    print(f"  var  = {var:.17g}")
    print(f"  std  = {std:.17g}")
    print(f"  source = {stats_source}")
    print("\nEmissivity parameters:")
    print(f"  sigma_d = {sigma_d:.17g}, sigma_j = {sigma_j:.17g}")
    print(f"  jdisk0 = {jdisk0:.17g}, jjet0 = {jjet0:.17g}")
    print(f"  torus_r0 = {torus_r0:.17g}, torus_ar = {torus_ar:.17g}, torus_az = {torus_az:.17g}")
    print(f"  jet_width = {jet_width:.17g}, jet_opening = {jet_opening:.17g}, jet_side = {args.jet_side}")
    print(f"\nWrote:\n  {out_h5}\n  {out_xmf}")
    print("Open the .xmf file in VisIt.  Use the variable 'emissivity' for the lognormal source field.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
