#!/usr/bin/env python3
"""Integrate diagnostic trajectories for the inoisy4D disk four-velocity.

The script imports the equations from ``inoisy4d_geometry_preview.py`` so the
diagnostic stays tied to the same prescription used by the plotting helper and
the C solver.  It integrates the prescribed coordinate velocity

    dx^i/dt = u^i/u^t

rather than geodesics.  This is intentional: the purpose is to reveal what the
current inoisy4D advection field is doing away from the equatorial plane.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "inoisy4d_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "inoisy4d_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import inoisy4d_geometry_preview as geom

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    }
)


@dataclass(frozen=True)
class Case:
    name: str
    x: float
    y: float
    z: float
    t_end: float


def ell_limits(x: float, y: float, z: float, p: geom.Params) -> tuple[float, float]:
    """Return the circular timelike allowed range for ell=-u_phi/u_t."""

    bp = geom.bl_point(x, y, z, p)
    m = geom.kerr_metric(bp["r"], bp["sinth"], bp["costh"], p.spin)
    a = m["gphiphi_inv"]
    b = -2.0 * m["gtphi_inv"]
    c = m["gtt_inv"]
    disc = max(b * b - 4.0 * a * c, 0.0)
    if abs(a) < geom.SMALL:
        return -np.inf, np.inf
    l1 = (-b - math.sqrt(disc)) / (2.0 * a)
    l2 = (-b + math.sqrt(disc)) / (2.0 * a)
    return min(l1, l2), max(l1, l2)


def local_state(x: float, y: float, z: float, p: geom.Params) -> dict[str, float]:
    bp = geom.bl_point(x, y, z, p)
    m = geom.kerr_metric(bp["r"], bp["sinth"], bp["costh"], p.spin)
    ell = geom.ell_profile(max(bp["rho"], 1.0e-6), p)
    U_ell = geom.U_of_ell(ell, m)
    omega_raw = geom.omega_from_ell(ell, m)
    omega = geom.clamp_omega(omega_raw, m, p)
    hat_ur = geom.plunge_hat_ur(bp, m, p)
    free_ur = geom.freefall_ur(m)
    ur = p.betar * hat_ur + (1.0 - p.betar) * free_ur
    u, _ = geom.disk_four_velocity(x, y, z, p)
    v = geom.disk_velocity_cartesian(x, y, z, p)
    lmin, lmax = ell_limits(x, y, z, p)
    return {
        "r": bp["r"],
        "rho": bp["rho"],
        "ell": ell,
        "ell_min": lmin,
        "ell_max": lmax,
        "U_ell": U_ell,
        "ell_allowed": 1.0 if U_ell > geom.SMALL else 0.0,
        "omega_raw": omega_raw,
        "omega": omega,
        "hat_ur": hat_ur,
        "free_ur": free_ur,
        "ur": ur,
        "ut": float(u[0]),
        "speed": float(np.linalg.norm(v)),
    }


def velocity(xyz: np.ndarray, p: geom.Params) -> np.ndarray:
    v = geom.disk_velocity_cartesian(float(xyz[0]), float(xyz[1]), float(xyz[2]), p)
    if not np.all(np.isfinite(v)):
        return np.zeros(3)
    return v


def integrate_case(case: Case, p: geom.Params, dt: float, r_stop: float, max_abs: float) -> np.ndarray:
    n = max(2, int(math.ceil(case.t_end / dt)) + 1)
    path = np.empty((n, 4), dtype=float)
    xyz = np.array([case.x, case.y, case.z], dtype=float)
    t = 0.0
    path[0] = [t, *xyz]
    used = 1

    for i in range(1, n):
        h = min(dt, case.t_end - t)
        if h <= 0.0:
            break

        k1 = velocity(xyz, p)
        k2 = velocity(xyz + 0.5 * h * k1, p)
        k3 = velocity(xyz + 0.5 * h * k2, p)
        k4 = velocity(xyz + h * k3, p)
        xyz = xyz + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

        path[i] = [t, *xyz]
        used += 1

        r = float(np.linalg.norm(xyz))
        if r <= r_stop or np.any(np.abs(xyz) > max_abs):
            break

    return path[:used]


def default_cases(p: geom.Params) -> list[Case]:
    rms = geom.r_isco(p.spin, p.sigma)
    rh = geom.horizon_radius(p.spin)
    inside_eq = 0.5 * (rms + rh)
    near_axis = 0.30
    return [
        Case("equatorial outside ISCO", max(8.0, rms + 4.0), 0.0, 0.0, 220.0),
        Case("equatorial inside ISCO", inside_eq, 0.0, 0.0, 140.0),
        Case("off-plane, small rho outside ISCO sphere", inside_eq, 0.0, 5.0, 120.0),
        Case("high-z, small rho outside ISCO sphere", inside_eq, 0.0, 15.0, 100.0),
        Case("high-z, rho outside ISCO", max(5.0, rms + 2.0), 0.0, 15.0, 120.0),
        Case("near-axis sign-flip tube", near_axis, 0.0, 5.0, 80.0),
    ]


def plot_trajectories(paths: list[np.ndarray], cases: list[Case], p: geom.Params, outfile: Path) -> None:
    fig = plt.figure(figsize=(9.5, 7.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(paths)))
    for path, case, color in zip(paths, cases, colors):
        ax.plot(path[:, 1], path[:, 2], path[:, 3], lw=1.8, color=color, label=case.name)
        ax.scatter(path[0, 1], path[0, 2], path[0, 3], color=color, marker="o", s=22)
        ax.scatter(path[-1, 1], path[-1, 2], path[-1, 3], color=color, marker="x", s=32)

    rms = geom.r_isco(p.spin, p.sigma)
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(rms * np.cos(theta), rms * np.sin(theta), 0.0 * theta, "k--", lw=1.0, label="equatorial ISCO")

    lim = 18.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-2.0, lim)
    ax.set_xlabel("x [M]")
    ax.set_ylabel("y [M]")
    ax.set_zlabel("z [M]")
    ax.view_init(elev=23.0, azim=-58.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.set_title("Prescribed disk four-velocity trajectories")
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rhoz_paths(paths: list[np.ndarray], cases: list[Case], p: geom.Params, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(paths)))
    for path, case, color in zip(paths, cases, colors):
        rho = np.hypot(path[:, 1], path[:, 2])
        ax.plot(rho, path[:, 3], color=color, lw=1.8, label=case.name)
        ax.scatter(rho[0], path[0, 3], color=color, marker="o", s=18)
        ax.scatter(rho[-1], path[-1, 3], color=color, marker="x", s=30)

    rms = geom.r_isco(p.spin, p.sigma)
    rho_s = np.linspace(0.0, rms, 200)
    z_s = np.sqrt(np.maximum(rms * rms - rho_s * rho_s, 0.0))
    ax.plot(rho_s, z_s, color="k", ls="--", lw=1.0, label="r = r_ISCO")
    ax.set_xlabel("rho = sqrt(x^2+y^2) [M]")
    ax.set_ylabel("z [M]")
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(-1.0, 17.0)
    ax.set_title("Same trajectories in the rho-z plane")
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(outfile, dpi=220)
    plt.close(fig)


def plot_ell_diagnostic(p: geom.Params, outfile: Path, z_values: list[float]) -> None:
    rho = np.linspace(0.05, 12.0, 900)
    ell = np.array([geom.ell_profile(r, p) for r in rho])

    fig, axes = plt.subplots(len(z_values), 1, figsize=(7.8, 2.6 * len(z_values)), sharex=True, constrained_layout=True)
    if len(z_values) == 1:
        axes = [axes]

    for ax, z in zip(axes, z_values):
        lmins = []
        lmaxs = []
        omegas = []
        hats = []
        for r in rho:
            lmin, lmax = ell_limits(float(r), 0.0, z, p)
            st = local_state(float(r), 0.0, z, p)
            lmins.append(lmin)
            lmaxs.append(lmax)
            omegas.append(st["omega"])
            hats.append(st["hat_ur"])

        lmins = np.array(lmins)
        lmaxs = np.array(lmaxs)
        omegas = np.array(omegas)
        hats = np.array(hats)

        uvals = []
        for r in rho:
            bp = geom.bl_point(float(r), 0.0, z, p)
            m = geom.kerr_metric(bp["r"], bp["sinth"], bp["costh"], p.spin)
            uvals.append(geom.U_of_ell(geom.ell_profile(float(r), p), m))
        uvals = np.array(uvals)

        ax.plot(rho, lmins, color="0.55", lw=1.0, ls="--", label="ell roots")
        ax.plot(rho, lmaxs, color="0.55", lw=1.0, ls="--")
        ax.plot(rho, ell, color="tab:blue", lw=1.6, label="ell_profile(rho)")
        ax.axhline(0.0, color="0.25", lw=0.8)
        rms = geom.r_isco(p.spin, p.sigma)
        if abs(z) < rms:
            ax.axvline(np.sqrt(max(rms * rms - z * z, 0.0)), color="k", ls="--", lw=1.0)
        active = np.abs(hats) > 1.0e-10
        if np.any(active):
            ax.fill_between(rho, 0.0, 1.0, where=active, color="tab:red", alpha=0.08, transform=ax.get_xaxis_transform())
        invalid = uvals <= geom.SMALL
        if np.any(invalid):
            ax.fill_between(rho, 0.0, 1.0, where=invalid, color="tab:purple", alpha=0.10, transform=ax.get_xaxis_transform())
        ax.set_ylabel(f"ell, z={z:g}")
        ax.set_ylim(-8.0, 8.0)
        ax.grid(alpha=0.18)

        twin = ax.twinx()
        twin.plot(rho, omegas, color="tab:orange", alpha=0.65, lw=1.0, label="Omega")
        twin.set_ylabel("Omega")
        twin.set_ylim(-1.2, 1.2)

    axes[-1].set_xlabel("rho [M]")
    axes[0].legend(loc="upper right")
    fig.savefig(outfile, dpi=220)
    plt.close(fig)


def plot_plunge_map(p: geom.Params, outfile: Path) -> None:
    rho = np.linspace(0.05, 8.0, 180)
    z = np.linspace(0.0, 18.0, 180)
    RHO, Z = np.meshgrid(rho, z, indexing="xy")
    HAT = np.empty_like(RHO)
    OMEGA = np.empty_like(RHO)

    for idx in np.ndindex(RHO.shape):
        st = local_state(float(RHO[idx]), 0.0, float(Z[idx]), p)
        HAT[idx] = st["hat_ur"]
        OMEGA[idx] = st["omega"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    im0 = ax0.imshow(
        HAT,
        origin="lower",
        extent=(rho.min(), rho.max(), z.min(), z.max()),
        aspect="auto",
        cmap="magma_r",
    )
    fig.colorbar(im0, ax=ax0, label="hat u^r plunge term")
    rms = geom.r_isco(p.spin, p.sigma)
    rho_s = np.linspace(0.0, rms, 200)
    z_s = np.sqrt(np.maximum(rms * rms - rho_s * rho_s, 0.0))
    ax0.plot(rho_s, z_s, color="cyan", ls="--", lw=1.0)
    ax0.set_xlabel("rho [M]")
    ax0.set_ylabel("z [M]")
    ax0.set_title("Where the spherical inner radial term is nonzero")

    im1 = ax1.imshow(
        OMEGA,
        origin="lower",
        extent=(rho.min(), rho.max(), z.min(), z.max()),
        aspect="auto",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.colorbar(im1, ax=ax1, label="clamped Omega")
    ax1.plot(rho_s, z_s, color="k", ls="--", lw=1.0)
    ax1.set_xlabel("rho [M]")
    ax1.set_ylabel("z [M]")
    ax1.set_title("Angular velocity after timelike clamp")

    fig.savefig(outfile, dpi=220)
    plt.close(fig)


def write_summary(cases: list[Case], p: geom.Params, outfile: Path) -> None:
    lines = []
    lines.append("# inoisy4D disk four-velocity trajectory diagnostic")
    lines.append("")
    lines.append(f"spin = {p.spin:.6g}")
    lines.append(f"sigma = {p.sigma:d}")
    lines.append(f"r_horizon = {geom.horizon_radius(p.spin):.8g}")
    lines.append(f"r_ISCO = {geom.r_isco(p.spin, p.sigma):.8g}")
    lines.append(f"betar = {p.betar:.6g}")
    lines.append(f"use_plunge = {p.use_plunge:d}")
    lines.append(f"enforce_timelike = {p.enforce_timelike:d}")
    lines.append("")
    lines.append("| case | rho0 | z0 | ell0 | ell roots | U(ell0) | Omega0 | hat_ur0 | u^t0 | speed0 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        st = local_state(case.x, case.y, case.z, p)
        lines.append(
            "| "
            + " | ".join(
                [
                    case.name,
                    f"{st['rho']:.6g}",
                    f"{case.z:.6g}",
                    f"{st['ell']:.6g}",
                    f"[{st['ell_min']:.6g}, {st['ell_max']:.6g}]",
                    f"{st['U_ell']:.6g}",
                    f"{st['omega']:.6g}",
                    f"{st['hat_ur']:.6g}",
                    f"{st['ut']:.6g}",
                    f"{st['speed']:.6g}",
                ]
            )
            + " |"
        )
    outfile.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=str(script_dir / "params_torusjet.h5"), help="HDF5 parameter file")
    ap.add_argument("--outdir", default=str(script_dir / "trajectory_diagnostic"), help="output directory")
    ap.add_argument("--dt", type=float, default=0.05, help="coordinate-time integration step")
    ap.add_argument("--max-abs", type=float, default=60.0, help="stop if any coordinate exceeds this absolute value")
    ap.add_argument("--z-ell", type=float, nargs="*", default=[0.0, 5.0, 15.0], help="z slices for ell diagnostic")
    args = ap.parse_args()

    p = geom.read_params(args.params)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = default_cases(p)
    r_stop = geom.horizon_radius(p.spin) + 1.0e-4
    paths = [integrate_case(case, p, args.dt, r_stop, args.max_abs) for case in cases]

    plot_trajectories(paths, cases, p, outdir / "disk_trajectory_3d.png")
    plot_rhoz_paths(paths, cases, p, outdir / "disk_trajectory_rhoz.png")
    plot_ell_diagnostic(p, outdir / "ell_axis_diagnostic.png", args.z_ell)
    plot_plunge_map(p, outdir / "plunge_and_omega_map.png")
    write_summary(cases, p, outdir / "summary.md")

    print(f"wrote {outdir / 'disk_trajectory_3d.png'}")
    print(f"wrote {outdir / 'disk_trajectory_rhoz.png'}")
    print(f"wrote {outdir / 'ell_axis_diagnostic.png'}")
    print(f"wrote {outdir / 'plunge_and_omega_map.png'}")
    print(f"wrote {outdir / 'summary.md'}")


if __name__ == "__main__":
    main()
