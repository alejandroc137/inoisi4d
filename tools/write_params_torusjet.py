#!/usr/bin/env python3
import math
import h5py

outfile = "params_torusjet.h5"

params_f64 = {
    # Disk/torus GRF correlation lengths.
    # Use a positive lam0 for practical runs.  lam0 <= 0 switches to
    # 2*pi/|Omega|, which can be stiff on a large box.
    "lam0": -1.0,
    "lam1": 5.0,
    "lam2": 1.5,
    "lam3": 0.6,

    # Jet GRF correlation lengths.
    "jet_lam0": 10.0,
    "jet_lam1": 2.5,
    "jet_lam2": 2.5,
    "jet_lam3": 2.5,

    # Surface pitch on the torus and helical pitch in the jet.
    "pitch": math.pi / 20.0,
    "jet_chi0": math.pi / 20.0,
    "jet_chi_rho": 2.0,

    # Torus-only correlation geometry.  These do not change the Kerr velocity.
    "torus_r0": 12.0,
    "torus_ar": 8.0,
    "torus_az": 5.0,

    # Jet-only correlation geometry.  Width(z)=jet_width+jet_opening*abs(z).
    "jet_width": 2.5,
    "jet_opening": 0.30,

    # In component=2, weights enter the single tensor as
    # Lambda = wd Lambda_disk + wj Lambda_jet.
    "blend_floor": 1.0e-4,

    # Kerr/off-equatorial disk four-velocity parameters.
    "spin": 0.94,
    "xi": 1.0,
    "delta": 0.0,
    "betar": 0.7,

    # Simple jet advection model.
    "jet_vpol": 0.5,
    "jet_omega": 0.0,
}

params_i32 = {
    # 0=disk/torus only, 1=jet only, 2=single composite torus+jet tensor.
    "component": 2,

    # +1 prograde, -1 retrograde.
    "sigma": 1,

    # Optional frozen-ISCO plunge branch.
    "use_plunge": 1,

    # Clamp Omega into the local Kerr timelike cone if necessary.
    "enforce_timelike": 1,

    # 0: weights localize the raw field; 1: weights only interpolate directions.
    "normalize_weights": 0,
}

with h5py.File(outfile, "w") as f:
    g = f.create_group("params")
    for name, value in params_f64.items():
        g.create_dataset(name, data=float(value))
    for name, value in params_i32.items():
        g.create_dataset(name, data=int(value))

print(f"wrote {outfile}")
