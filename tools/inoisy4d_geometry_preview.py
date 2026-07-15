#!/usr/bin/env python3
"""Preview the inoisy+ four-velocity and correlation tensor.

This script mirrors the equations implemented in src/param_inoisy4d.c.  It is
intended for rapid design of the disk/torus and jet correlation geometry before
running the HYPRE solver.

Examples
--------
python3 inoisy4d_geometry_preview.py \
  --plane xy \
  --vector diskv \
  --background weights \
  --arrow-mode raw \
  --out xy_disk_speed.png


python3 inoisy4d_geometry_preview.py \
  --plane xz \
  --vector jetv \
  --background weights \
  --arrow-mode raw \
  --out xz_jet_speed.png
  """

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size' : 14,                   # Set font size to 11pt
    'axes.labelsize': 14,               # -> axis labels
    'legend.fontsize': 12,              # -> legends
    'text.usetex': True,
    'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage{lmodern}'
    ),
    'font.family': 'Latin Modern Roman',
})

SMALL = 1.0e-12
OMEGA_FLOOR = 1.0e-8
LAMBDA_FLOOR = 1.0e-8


@dataclass
class Params:
    component: int = 2

    lam0: float = -1.0
    lam1: float = 5.0
    lam2: float = 1.5
    lam3: float = 0.6

    jet_lam0: float = 10.0
    jet_lam1: float = 2.5
    jet_lam2: float = 2.5
    jet_lam3: float = 2.5

    pitch: float = np.pi / 20.0
    jet_chi0: float = np.pi / 20.0
    jet_chi_rho: float = 2.0

    torus_r0: float = 12.0
    torus_ar: float = 8.0
    torus_az: float = 5.0
    jet_width: float = 2.5
    jet_opening: float = 0.30
    blend_floor: float = 1.0e-4
    normalize_weights: int = 0

    spin: float = 0.94
    xi: float = 1.0
    delta: float = 0.0
    sigma: int = 1
    betar: float = 0.7
    use_plunge: int = 0
    enforce_timelike: int = 1

    jet_vpol: float = 0.5
    jet_omega: float = 0.0 


def read_params(filename: str | None) -> Params:
    p = Params()
    if filename is None:
        return p
    names = {f.name for f in fields(p)}
    with h5py.File(filename, "r") as f:
        g = f["params"]
        for name in names:
            if name in g:
                setattr(p, name, g[name][()].item())
    p.component = int(p.component)
    p.sigma = 1 if int(p.sigma) >= 0 else -1
    p.spin = float(np.clip(p.spin, -0.999999, 0.999999))
    p.betar = float(np.clip(p.betar, 0.0, 1.0))
    p.torus_r0 = positive(p.torus_r0)
    p.torus_ar = positive(p.torus_ar)
    p.torus_az = positive(p.torus_az)
    p.jet_width = positive(p.jet_width)
    p.jet_opening = max(0.0, float(p.jet_opening))
    p.blend_floor = max(0.0, float(p.blend_floor))
    p.normalize_weights = 1 if int(p.normalize_weights) else 0
    return p


def positive(x: float) -> float:
    if not np.isfinite(x):
        return LAMBDA_FLOOR
    return max(abs(float(x)), LAMBDA_FLOOR)


def horizon_radius(a: float) -> float:
    return 1.0 + np.sqrt(max(0.0, 1.0 - a * a))


def bl_point(x: float, y: float, z: float, p: Params):
    r_raw = np.sqrt(x * x + y * y + z * z)
    rho = np.sqrt(x * x + y * y)
    r_min = horizon_radius(p.spin) + 1.0e-6
    if r_raw > SMALL:
        sinth = rho / r_raw
        costh = z / r_raw
        xnorm = x / r_raw
        ynorm = y / r_raw
        znorm = z / r_raw
    else:
        sinth = 1.0
        costh = 0.0
        xnorm = ynorm = znorm = 0.0
    return {
        "r": max(r_raw, r_min),
        "rho": rho,
        "sinth": sinth,
        "costh": costh,
        "phi": np.arctan2(y, x),
        "xnorm": xnorm,
        "ynorm": ynorm,
        "znorm": znorm,
    }


def kerr_metric(r: float, sinth: float, costh: float, a: float):
    sin2 = max(sinth * sinth, 1.0e-14)
    cos2 = costh * costh
    Sigma = r * r + a * a * cos2
    Delta = max(r * r - 2.0 * r + a * a, 1.0e-14)
    A = (r * r + a * a) ** 2 - a * a * Delta * sin2
    return {
        "Delta": Delta,
        "Sigma": Sigma,
        "A": A,
        "gtt": -(1.0 - 2.0 * r / Sigma),
        "gtphi": -2.0 * a * r * sin2 / Sigma,
        "grr": Sigma / Delta,
        "gphiphi": A * sin2 / Sigma,
        "gtt_inv": -A / (Sigma * Delta),
        "gtphi_inv": -2.0 * a * r / (Sigma * Delta),
        "grr_inv": Delta / Sigma,
        "gphiphi_inv": (Delta - a * a * sin2) / (Sigma * Delta * sin2),
    }


def r_isco(a: float, sigma: int) -> float:
    aa = abs(a)
    z1 = 1.0 + (1.0 - aa * aa) ** (1.0 / 3.0) * (
        (1.0 + aa) ** (1.0 / 3.0) + (1.0 - aa) ** (1.0 / 3.0)
    )
    z2 = np.sqrt(3.0 * aa * aa + z1 * z1)
    branch = -1.0 if sigma >= 0 else 1.0
    return 3.0 + z2 + branch * np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def ell_profile(rho: float, p: Params) -> float:
    r = max(float(rho), 1.0e-6)
    sr = np.sqrt(r)
    sigma = 1.0 if p.sigma >= 0 else -1.0
    a = p.spin
    denom = r - 2.0 + p.delta + sigma * a / sr
    if abs(denom) < 1.0e-10:
        denom = 1.0e-10 if denom >= 0.0 else -1.0e-10
    return p.xi * sigma * (r * sr - 2.0 * sigma * a + a * a / sr) / denom


def U_of_ell(ell: float, m) -> float:
    return -m["gtt_inv"] + 2.0 * ell * m["gtphi_inv"] - ell * ell * m["gphiphi_inv"]


def omega_from_ell(ell: float, m) -> float:
    denom = m["gtt_inv"] - ell * m["gtphi_inv"]
    if abs(denom) < SMALL:
        return 0.0
    return (m["gtphi_inv"] - ell * m["gphiphi_inv"]) / denom


def clamp_omega(omega: float, m, p: Params) -> float:
    if not p.enforce_timelike or m["gphiphi"] <= SMALL:
        return omega
    disc = max(m["gtphi"] * m["gtphi"] - m["gtt"] * m["gphiphi"], 0.0)
    disc = np.sqrt(disc)
    om1 = (-m["gtphi"] - disc) / m["gphiphi"]
    om2 = (-m["gtphi"] + disc) / m["gphiphi"]
    omin, omax = min(om1, om2), max(om1, om2)
    eps = 1.0e-8 * max(1.0, omax - omin)
    return float(np.clip(omega, omin + eps, omax - eps))


def D_of_omega(omega: float, m) -> float:
    return -m["gtt"] - 2.0 * m["gtphi"] * omega - m["gphiphi"] * omega * omega


def freefall_ur(m) -> float:
    return -np.sqrt(max(0.0, (-1.0 - m["gtt_inv"]) * m["grr_inv"]))


def plunge_hat_ur(bp, m, p: Params) -> float:
    if not p.use_plunge:
        return 0.0
    rms = r_isco(p.spin, p.sigma)
    if bp["r"] >= rms:
        return 0.0
    rb = rms
    sinb = bp["sinth"]
    cosb = bp["costh"]
    rho_b = rb * sinb
    ell_b = ell_profile(rho_b, p)
    mb = kerr_metric(rb, sinb, cosb, p.spin)
    Ub = U_of_ell(ell_b, mb)
    if Ub <= SMALL or not np.isfinite(Ub):
        return 0.0
    Eb = 1.0 / np.sqrt(Ub)
    Lb = Eb * ell_b
    Rpl = (
        -1.0
        - m["gtt_inv"] * Eb * Eb
        + 2.0 * m["gtphi_inv"] * Eb * Lb
        - m["gphiphi_inv"] * Lb * Lb
    )
    if Rpl <= 0.0 or not np.isfinite(Rpl):
        return 0.0
    return -np.sqrt(max(0.0, m["grr_inv"] * Rpl))


def disk_four_velocity(x: float, y: float, z: float, p: Params):
    bp = bl_point(x, y, z, p)
    m = kerr_metric(bp["r"], bp["sinth"], bp["costh"], p.spin)
    ell = ell_profile(max(bp["rho"], 1.0e-6), p)
    omega = clamp_omega(omega_from_ell(ell, m), m, p)
    hat_ur = plunge_hat_ur(bp, m, p)
    bar_ur = freefall_ur(m)
    ur = p.betar * hat_ur + (1.0 - p.betar) * bar_ur
    D = D_of_omega(omega, m)
    if D <= SMALL or not np.isfinite(D):
        omega = 0.0
        ur = 0.0
        D = D_of_omega(omega, m)
    ut = np.sqrt(max(SMALL, (1.0 + m["grr"] * ur * ur) / max(D, SMALL)))
    uphi = omega * ut
    return np.array([ut, ur, 0.0, uphi]), omega


def disk_velocity_cartesian(x: float, y: float, z: float, p: Params):
    bp = bl_point(x, y, z, p)
    u, omega = disk_four_velocity(x, y, z, p)
    vr = u[1] / u[0]
    return np.array([vr * bp["xnorm"] - omega * y,
                     vr * bp["ynorm"] + omega * x,
                     vr * bp["znorm"]])


def jet_velocity_cartesian(x: float, y: float, z: float, p: Params):
    r = np.sqrt(x * x + y * y + z * z)
    er = np.array([0.0, 0.0, 1.0 if z >= 0.0 else -1.0])
    if r > SMALL:
        er = np.array([x / r, y / r, z / r])
    return np.array([p.jet_vpol * er[0] - p.jet_omega * y,
                     p.jet_vpol * er[1] + p.jet_omega * x,
                     p.jet_vpol * er[2]])


def q0_from_v(v):
    return np.array([1.0, v[0], v[1], v[2]])


def cylindrical_frame(x: float, y: float):
    rho = np.sqrt(x * x + y * y)
    erho = np.array([1.0, 0.0, 0.0])
    ephi = np.array([0.0, 1.0, 0.0])
    if rho > SMALL:
        erho = np.array([x / rho, y / rho, 0.0])
        ephi = np.array([-y / rho, x / rho, 0.0])
    return erho, ephi


def normalize3(v, fallback):
    n = np.linalg.norm(v)
    if n > SMALL and np.isfinite(n):
        return v / n
    return np.array(fallback, dtype=float)


def torus_weight(x: float, y: float, z: float, p: Params) -> float:
    rho = np.sqrt(x * x + y * y)
    u = (rho - p.torus_r0) / max(p.torus_ar, LAMBDA_FLOOR)
    v = z / max(p.torus_az, LAMBDA_FLOOR)
    return float(np.exp(-0.5 * (u * u + v * v)))


def jet_weight(x: float, y: float, z: float, p: Params) -> float:
    rho = np.sqrt(x * x + y * y)
    width = max(p.jet_width + p.jet_opening * abs(z), LAMBDA_FLOOR)
    return float(np.exp(-0.5 * rho * rho / (width * width)))


def composite_weights(x: float, y: float, z: float, p: Params):
    d = torus_weight(x, y, z, p)
    j = jet_weight(x, y, z, p)
    if p.normalize_weights:
        s = max(d + j + 2.0 * p.blend_floor, SMALL)
        return (d + p.blend_floor) / s, (j + p.blend_floor) / s
    return d + p.blend_floor, j + p.blend_floor


def disk_triad(x: float, y: float, z: float, p: Params):
    rho = np.sqrt(x * x + y * y)
    u = (rho - p.torus_r0) / max(p.torus_ar, LAMBDA_FLOOR)
    v = z / max(p.torus_az, LAMBDA_FLOOR)
    erho, ephi = cylindrical_frame(x, y)
    epol = normalize3(-p.torus_ar * v * erho + np.array([0.0, 0.0, p.torus_az * u]), [0.0, 0.0, 1.0])
    enorm = normalize3(p.torus_az * u * erho + np.array([0.0, 0.0, p.torus_ar * v]), erho)
    cp, sp = np.cos(p.pitch), np.sin(p.pitch)
    e1 = cp * ephi + sp * epol
    e2 = -sp * ephi + cp * epol
    e3 = enorm
    return np.array([0.0, *e1]), np.array([0.0, *e2]), np.array([0.0, *e3])


def jet_triad(x: float, y: float, z: float, p: Params):
    rho = np.sqrt(x * x + y * y)
    r = np.sqrt(rho * rho + z * z)
    er = np.array([0.0, 0.0, 1.0 if z >= 0.0 else -1.0])
    eth = np.array([1.0, 0.0, 0.0])
    eph = np.array([0.0, 1.0, 0.0])
    if rho > SMALL:
        eph = np.array([-y / rho, x / rho, 0.0])
    if r > SMALL and rho > SMALL:
        er = np.array([x / r, y / r, z / r])
        eth = np.array([x * z / (rho * r), y * z / (rho * r), -rho / r])
    chi = p.jet_chi0 * (1.0 - np.exp(-rho * rho / max(p.jet_chi_rho * p.jet_chi_rho, SMALL)))
    cc, sc = np.cos(chi), np.sin(chi)
    e1 = cc * er + sc * eph
    e2 = eth
    e3 = -sc * er + cc * eph
    return np.array([0.0, *e1]), np.array([0.0, *e2]), np.array([0.0, *e3])


def disk_lambdas(x: float, y: float, z: float, p: Params):
    if p.lam0 > 0.0:
        lam0 = positive(p.lam0)
    else:
        _, omega = disk_four_velocity(x, y, z, p)
        lam0 = positive(2.0 * np.pi / max(abs(omega), OMEGA_FLOOR))
    return np.array([lam0, positive(p.lam1), positive(p.lam2), positive(p.lam3)])


def jet_lambdas(p: Params):
    return np.array([positive(p.jet_lam0), positive(p.jet_lam1), positive(p.jet_lam2), positive(p.jet_lam3)])


def tensor_component(weight, lambdas, q0, q1, q2, q3):
    qs = [q0, q1, q2, q3]
    L = np.zeros((4, 4))
    for lam, q in zip(lambdas, qs):
        L += weight * lam * lam * np.outer(q, q)
    return L


def lambda_tensor(x: float, y: float, z: float, p: Params):
    L = np.zeros((4, 4))
    if p.component in (0, 2):
        wd = 1.0
        if p.component == 2:
            wd, _ = composite_weights(x, y, z, p)
        q0 = q0_from_v(disk_velocity_cartesian(x, y, z, p))
        q1, q2, q3 = disk_triad(x, y, z, p)
        L += tensor_component(wd, disk_lambdas(x, y, z, p), q0, q1, q2, q3)
    if p.component in (1, 2):
        wj = 1.0
        if p.component == 2:
            _, wj = composite_weights(x, y, z, p)
        q0 = q0_from_v(jet_velocity_cartesian(x, y, z, p))
        q1, q2, q3 = jet_triad(x, y, z, p)
        L += tensor_component(wj, jet_lambdas(p), q0, q1, q2, q3)
    return L


def vector_for_plot(x: float, y: float, z: float, p: Params, name: str):
    if name == "disk1":
        return disk_triad(x, y, z, p)[0][1:]
    if name == "disk2":
        return disk_triad(x, y, z, p)[1][1:]
    if name == "disk3":
        return disk_triad(x, y, z, p)[2][1:]
    if name == "jet1":
        return jet_triad(x, y, z, p)[0][1:]
    if name == "jet2":
        return jet_triad(x, y, z, p)[1][1:]
    if name == "jet3":
        return jet_triad(x, y, z, p)[2][1:]
    if name == "diskv":
        return disk_velocity_cartesian(x, y, z, p)
    if name == "jetv":
        return jet_velocity_cartesian(x, y, z, p)
    if name == "principal":
        S = lambda_tensor(x, y, z, p)[1:, 1:]
        vals, vecs = np.linalg.eigh(0.5 * (S + S.T))
        return vecs[:, np.argmax(vals)]
    raise ValueError(f"unknown vector '{name}'")


def background_value(x: float, y: float, z: float, p: Params, name: str, vector: str | None = None):
    if name == "disk_weight":
        return torus_weight(x, y, z, p)
    if name == "jet_weight":
        return jet_weight(x, y, z, p)
    if name == "weights":
        wd, wj = composite_weights(x, y, z, p)
        return wd + wj
    if name == "detlambda":
        return max(np.linalg.det(lambda_tensor(x, y, z, p)), 0.0) ** 0.25
    if name == "omega":
        return disk_four_velocity(x, y, z, p)[1]
    if name == "disk_speed":
        return float(np.linalg.norm(disk_velocity_cartesian(x, y, z, p)))
    if name == "jet_speed":
        return float(np.linalg.norm(jet_velocity_cartesian(x, y, z, p)))
    if name == "vector_norm":
        if vector is None:
            raise ValueError("background='vector_norm' requires a plotted vector")
        return float(np.linalg.norm(vector_for_plot(x, y, z, p, vector)))
    raise ValueError(f"unknown background '{name}'")


def make_grid(plane: str, extent: float, n: int, fixed: float):
    a = np.linspace(-extent, extent, n)
    b = np.linspace(-extent, extent, n)
    A, B = np.meshgrid(a, b, indexing="xy")
    if plane == "xy":
        X, Y, Z = A, B, np.full_like(A, fixed)
        labels = (r"$x_{\rm{s}}$", r"$y_{\rm{s}}$")
        proj = (0, 1)
    elif plane == "xz":
        X, Y, Z = A, np.full_like(A, fixed), B
        labels = (r"$x_{\rm{s}}$", r"$z_{\rm{s}}$")
        proj = (0, 2)
    elif plane == "yz":
        X, Y, Z = np.full_like(A, fixed), A, B
        labels = (r"$y_{\rm{s}}$", r"$z_{\rm{s}}$")
        proj = (1, 2)
    else:
        raise ValueError("plane must be xy, xz, or yz")
    return A, B, X, Y, Z, labels, proj




def transform_vector_for_plot(vec3: np.ndarray, proj: tuple[int, int], mode: str) -> tuple[float, float]:
    """Return the two displayed vector components.

    mode='raw' keeps the physical/projected magnitude.
    mode='unit3d' normalizes the full 3D vector before projection. This was
    the behavior of the original preview script.
    mode='unit2d' projects first and then normalizes in the plotted plane.
    """
    v = np.array(vec3, dtype=float, copy=True)
    if mode == "none":
        mode = "raw"

    if mode == "unit3d":
        n3 = np.linalg.norm(v)
        if n3 > SMALL and np.isfinite(n3):
            v = v / n3
        return float(v[proj[0]]), float(v[proj[1]])

    u = np.array([v[proj[0]], v[proj[1]]], dtype=float)
    if mode == "unit2d":
        n2 = np.linalg.norm(u)
        if n2 > SMALL and np.isfinite(n2):
            u = u / n2
    elif mode != "raw":
        raise ValueError(f"unknown arrow_mode '{mode}'")
    return float(u[0]), float(u[1])


def default_arrow_mode(vector: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if vector in ("diskv", "jetv"):
        return "raw"
    return "unit2d"

def plot_preview(p: Params, plane: str, vector: str, background: str, extent: float, n: int,
                 stride: int, fixed: float, outfile: str, arrow_mode: str, quiver_scale: float | None):
    A, B, X, Y, Z, labels, proj = make_grid(plane, extent, n, fixed)
    bg = np.empty_like(A)
    U = np.empty_like(A)
    V = np.empty_like(A)

    mode = default_arrow_mode(vector, arrow_mode)
    norm_min = np.inf
    norm_max = -np.inf

    for idx in np.ndindex(A.shape):
        x, y, z = float(X[idx]), float(Y[idx]), float(Z[idx])
        vec = vector_for_plot(x, y, z, p, vector)
        bg[idx] = background_value(x, y, z, p, background, vector=vector)
        norm = float(np.linalg.norm(vec))
        norm_min = min(norm_min, norm)
        norm_max = max(norm_max, norm)
        U[idx], V[idx] = transform_vector_for_plot(vec, proj, mode)

    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    im = ax.imshow(bg, origin="lower", extent=(-extent, extent, -extent, extent), aspect="equal",alpha=0.7)
    fig.colorbar(im, ax=ax, label=background)
    s = max(1, int(stride))
    quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
    if quiver_scale is not None:
        quiver_kwargs["scale"] = quiver_scale
    ax.quiver(A[::s, ::s], B[::s, ::s], U[::s, ::s], V[::s, ::s], **quiver_kwargs)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    #ax.set_title(f"{plane} slice, vector={vector}, background={background}")
    fig.savefig(outfile, dpi=180)
    print(f"wrote {outfile}")
    print(f"vector={vector}, arrow_mode={mode}, |vector| range = [{norm_min:.6g}, {norm_max:.6g}]")

def plot_xy_xz_shared_x(
    p: Params,
    vector_xy: str = "diskv",
    vector_xz: str = "jetv",
    background: str = "weights",
    extent: float = 50.0,
    n: int = 81,
    stride: int = 5,
    fixed_xy: float = 0.0,
    fixed_xz: float = 0.0,
    outfile: str = "xy_xz_weights.png",
    arrow_mode: str = "auto",
    quiver_scale: float | None = None,
    figsize: tuple[float, float] = (7.0, 9.0),
    alpha: float = 0.75,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
):
    """Plot xy and xz slices in one figure with a shared x axis and one colorbar.

    Top panel:    xy plane at z=fixed_xy.
    Bottom panel: xz plane at y=fixed_xz.

    The background is evaluated independently on both planes, but one common
    color scale is used.  The default background is W_disk + W_jet.
    """

    def _background_value_safe(x, y, z, vector_name):
        try:
            return background_value(x, y, z, p, background, vector=vector_name)
        except TypeError:
            return background_value(x, y, z, p, background)

    def _panel_data(plane: str, vector_name: str, fixed: float):
        A, B, X, Y, Z, labels, proj = make_grid(plane, extent, n, fixed)
        bg = np.empty_like(A)
        U = np.empty_like(A)
        V = np.empty_like(A)

        mode = default_arrow_mode(vector_name, arrow_mode)
        norm_min = np.inf
        norm_max = -np.inf

        for idx in np.ndindex(A.shape):
            x, y, z = float(X[idx]), float(Y[idx]), float(Z[idx])

            vec = vector_for_plot(x, y, z, p, vector_name)
            bg[idx] = _background_value_safe(x, y, z, vector_name)

            norm = float(np.linalg.norm(vec))
            norm_min = min(norm_min, norm)
            norm_max = max(norm_max, norm)

            U[idx], V[idx] = transform_vector_for_plot(vec, proj, mode)

        return A, B, bg, U, V, labels, mode, norm_min, norm_max

    Axy, Bxy, bg_xy, Uxy, Vxy, labels_xy, mode_xy, nmin_xy, nmax_xy = _panel_data(
        "xy", vector_xy, fixed_xy
    )
    Axz, Bxz, bg_xz, Uxz, Vxz, labels_xz, mode_xz, nmin_xz, nmax_xz = _panel_data(
        "xz", vector_xz, fixed_xz
    )

    if vmin is None:
        vmin = min(float(np.nanmin(bg_xy)), float(np.nanmin(bg_xz)))
    if vmax is None:
        vmax = max(float(np.nanmax(bg_xy)), float(np.nanmax(bg_xz)))

    fig, (ax_xy, ax_xz) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )

    im_kwargs = dict(
        origin="lower",
        extent=(-extent, extent, -extent, extent),
        aspect="equal",
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )
    if cmap is not None:
        im_kwargs["cmap"] = cmap

    im = ax_xy.imshow(bg_xy, **im_kwargs)
    ax_xz.imshow(bg_xz, **im_kwargs)

    s = max(1, int(stride))
    quiver_kwargs = {
        "angles": "xy",
        "scale_units": "xy",
        "pivot": "middle",
    }
    if quiver_scale is not None:
        quiver_kwargs["scale"] = quiver_scale

    ax_xy.quiver(
        Axy[::s, ::s],
        Bxy[::s, ::s],
        Uxy[::s, ::s],
        Vxy[::s, ::s],
        **quiver_kwargs,
    )
    ax_xz.quiver(
        Axz[::s, ::s],
        Bxz[::s, ::s],
        Uxz[::s, ::s],
        Vxz[::s, ::s],
        **quiver_kwargs,
    )

    ax_xy.set_ylabel(labels_xy[1])
    ax_xz.set_xlabel(labels_xz[0])
    ax_xz.set_ylabel(labels_xz[1])

    ax_xy.tick_params(labelbottom=False)

    cbar = fig.colorbar(im, ax=(ax_xy, ax_xz), pad=0.02)
    cbar.set_label(colorbar_label or background)

    fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"wrote {outfile}")
    print(
        f"xy: vector={vector_xy}, arrow_mode={mode_xy}, "
        f"|vector| range=[{nmin_xy:.6g}, {nmax_xy:.6g}]"
    )
    print(
        f"xz: vector={vector_xz}, arrow_mode={mode_xz}, "
        f"|vector| range=[{nmin_xz:.6g}, {nmax_xz:.6g}]"
    )

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default=None, help="HDF5 parameter file with /params datasets")
    ap.add_argument("--plane", choices=["xy", "xz", "yz"], default="xz")
    ap.add_argument("--vector", choices=["principal", "disk1", "disk2", "disk3", "jet1", "jet2", "jet3", "diskv", "jetv"], default="principal")
    ap.add_argument("--background", choices=["weights", "disk_weight", "jet_weight", "detlambda", "omega",
                                             "disk_speed", "jet_speed", "vector_norm"], default="weights")
    ap.add_argument("--extent", type=float, default=50.0)
    ap.add_argument("--n", type=int, default=81)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fixed", type=float, default=0.0, help="fixed coordinate for the missing direction")
    ap.add_argument("--arrow-mode", choices=["auto", "raw", "unit3d", "unit2d", "none"], default="auto",
                    help="quiver scaling: raw magnitude, full-3D unit vectors, projected-2D unit vectors, or auto")
    ap.add_argument("--quiver-scale", type=float, default=None,
                    help="matplotlib quiver scale. Smaller values make arrows longer. Default lets matplotlib choose.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    p = read_params(args.params)
    out = args.out
    if out is None:
        stem = Path(args.params).stem if args.params else "defaults"
        out = f"{stem}_{args.plane}_{args.vector}_{args.background}.png"
    plot_preview(p, args.plane, args.vector, args.background, args.extent, args.n, args.stride, args.fixed,
                 out, args.arrow_mode, args.quiver_scale)


if __name__ == "__main__":
    main()


'''
Useful vector choices are:

principal   largest spatial eigenvector of Lambda
disk1       long torus-surface disk correlation direction
disk2       secondary torus-surface direction
disk3       torus cross-section normal
jet1        helical/poloidal jet direction
jet2        jet polar direction
jet3        complementary helical jet direction
diskv       disk advection velocity
jetv        jet advection velocity

Useful backgrounds are:

weights       W_disk + W_jet
disk_weight   torus weight only
jet_weight    jet weight only
detlambda     |Lambda|^{1/4}
omega         disk angular velocity

python3 - <<'PY'
import inoisy4d_geometry_preview as g

p = g.read_params(None)  # or g.read_params("params_torusjet.h5")

g.plot_xy_xz_shared_x(
    p,
    vector_xy="diskv",
    vector_xz="jetv",
    background="weights",
    extent=50.0,
    n=81,
    stride=5,
    fixed_xy=0.0,
    fixed_xz=0.0,
    arrow_mode="auto",
    quiver_scale=None,
    outfile="xy_xz_weights.png",
)
PY
'''
