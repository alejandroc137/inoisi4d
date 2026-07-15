#!/usr/bin/env python3
"""
Convert an inoisy4d/inoisy+ HDF5 output file into a VisIt-friendly
HDF5+XDMF pair.

The solver writes /data/data_raw with shape [Nt, Nx, Ny, Nz].  VisIt's
XDMF regular-mesh examples expect cell-centered scalar data in KJI order,
[Nz, Ny, Nx].  This script writes one transposed 3D dataset per selected
time step and an .xmf temporal collection that points to those datasets.

Usage:
    python3 inoisy4d_to_visit.py output.h5
    python3 inoisy4d_to_visit.py output.h5 -o visit_output --time-stride 4
    python3 inoisy4d_to_visit.py output.h5 -o preview --time-stride 4 --space-stride 2

Open the produced .xmf file in VisIt.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


def read_scalar(f: h5py.File, path: str, default=None):
    if path in f:
        return f[path][()]
    return default


def as_float(x, name: str) -> float:
    if x is None:
        raise RuntimeError(f"Missing required scalar {name!r} in input file")
    return float(np.asarray(x))


def write_xmf(
    xmf_path: Path,
    h5_name: str,
    dataset_names: list[str],
    time_values: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    var_name: str = "data_raw",
) -> None:
    ox, oy, oz = origin
    dx, dy, dz = spacing

    with xmf_path.open("w", encoding="utf-8") as xmf:
        xmf.write('<?xml version="1.0" ?>\n')
        xmf.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        xmf.write('<Xdmf Version="2.0">\n')
        xmf.write('  <Domain>\n')
        xmf.write('    <Grid Name="inoisy4d" GridType="Collection" CollectionType="Temporal">\n')

        for n, (dset, tval) in enumerate(zip(dataset_names, time_values)):
            xmf.write(f'      <Grid Name="step_{n:06d}" GridType="Uniform">\n')
            xmf.write(f'        <Time Value="{tval:.17g}"/>\n')
            # XDMF/VisIt structured-grid dimensions are written in KJI order.
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a VisIt-readable HDF5+XDMF time series from an inoisy4d output."
    )
    parser.add_argument("input", help="inoisy4d output .h5 file")
    parser.add_argument(
        "-o", "--output-prefix", default=None,
        help="output prefix; default is <input-stem>_visit"
    )
    parser.add_argument(
        "--dataset", default="/data/data_raw",
        help="input dataset path, default /data/data_raw"
    )
    parser.add_argument(
        "--time-stride", type=int, default=1,
        help="use every Nth time slice, default 1"
    )
    parser.add_argument(
        "--space-stride", type=int, default=1,
        help="use every Nth cell in each spatial direction, default 1"
    )
    parser.add_argument(
        "--first", type=int, default=0,
        help="first time index to export, default 0"
    )
    parser.add_argument(
        "--last", type=int, default=None,
        help="last time index to export, inclusive; default Nt-1"
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="gzip-compress the transposed HDF5 datasets"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing output files"
    )
    args = parser.parse_args()

    if args.time_stride < 1 or args.space_stride < 1:
        parser.error("--time-stride and --space-stride must be positive integers")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    prefix = args.output_prefix
    if prefix is None:
        prefix = str(input_path.with_name(input_path.stem + "_visit"))
    prefix_path = Path(prefix).resolve()

    out_h5 = prefix_path.with_suffix(".h5")
    out_xmf = prefix_path.with_suffix(".xmf")

    if not args.force:
        for p in (out_h5, out_xmf):
            if p.exists():
                raise FileExistsError(f"{p} exists; use --force to overwrite")

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
        last = nt - 1 if args.last is None else min(args.last, nt - 1)
        first = max(args.first, 0)
        if first > last:
            raise RuntimeError(f"No time slices selected: first={first}, last={last}, Nt={nt}")
        time_indices = list(range(first, last + 1, args.time_stride))

        sx = sy = sz = args.space_stride
        nx = len(range(0, nx0, sx))
        ny = len(range(0, ny0, sy))
        nz = len(range(0, nz0, sz))

        x0start = as_float(read_scalar(fin, "/params/x0start"), "/params/x0start")
        dx0 = as_float(read_scalar(fin, "/params/dx0"), "/params/dx0")
        x1start = as_float(read_scalar(fin, "/params/x1start"), "/params/x1start")
        x2start = as_float(read_scalar(fin, "/params/x2start"), "/params/x2start")
        x3start = as_float(read_scalar(fin, "/params/x3start"), "/params/x3start")
        dx1 = as_float(read_scalar(fin, "/params/dx1"), "/params/dx1") * sx
        dx2 = as_float(read_scalar(fin, "/params/dx2"), "/params/dx2") * sy
        dx3 = as_float(read_scalar(fin, "/params/dx3"), "/params/dx3") * sz

        time_values = np.array([x0start + dx0 * (it + 0.5) for it in time_indices], dtype=np.float64)
        compression = "gzip" if args.compress else None
        dataset_names: list[str] = []

        with h5py.File(out_h5, "w") as fout:
            gout = fout.create_group("data")
            # Copy scalar params for provenance.
            if "/params" in fin:
                fin.copy("/params", fout)
            else:
                fout.create_group("params")

            for n, it in enumerate(time_indices):
                # Original solver order: [t, x, y, z].
                # VisIt/XDMF regular cell data order: [z, y, x].
                arr_xyz = dset[it, ::sx, ::sy, ::sz]
                arr_zyx = np.asarray(arr_xyz).transpose(2, 1, 0)
                name = f"data_raw_{n:06d}"
                gout.create_dataset(name, data=arr_zyx, compression=compression)
                dpath = posixpath.join("/data", name)
                dataset_names.append(dpath)
                print(f"wrote {dpath} from input time index {it} / {nt - 1}")

            gout.create_dataset("time", data=time_values)
            fout["params"].attrs["visit_source_file"] = str(input_path)
            fout["params"].attrs["visit_source_dataset"] = args.dataset
            fout["params"].attrs["visit_original_shape_Nt_Nx_Ny_Nz"] = np.array(dset.shape, dtype=np.int64)
            fout["params"].attrs["visit_export_shape_Nt_Nx_Ny_Nz"] = np.array([len(time_indices), nx, ny, nz], dtype=np.int64)
            fout["params"].attrs["visit_time_stride"] = args.time_stride
            fout["params"].attrs["visit_space_stride"] = args.space_stride

    write_xmf(
        out_xmf,
        out_h5.name,
        dataset_names,
        time_values,
        nx=nx,
        ny=ny,
        nz=nz,
        origin=(x1start, x2start, x3start),
        spacing=(dx1, dx2, dx3),
    )

    print(f"\nWrote:\n  {out_h5}\n  {out_xmf}")
    print("Open the .xmf file in VisIt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
