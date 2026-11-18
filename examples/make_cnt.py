"""Write cnt.json: the four coils of CNT, the Columbia Non-neutral Torus, as Fourier filaments.

    python examples/make_cnt.py

Standard library only. CNT (Pedersen et al., Fusion Sci. Technol. 46, 200 (2004); 50, 372 (2006)) is a
two-period stellarator of ultralow aspect ratio made of four circular planar coils:

  two interlocked (IL) coils of radius R_c = 0.405 m, centres at x = +-0.313 m, their planes
  tilted against each other by 64 degrees, 170 kA each, and
  two poloidal field (PF) rings of radius 1.08 m at z = +-0.405 m, 40.3 kA each.

The IL coils follow the Fourier parameterisation of Yu et al., Phys. Plasmas 28, 092501 (2021):
  IL 1: (x_c0 + R_c cos t, -y_s1 sin t, -z_s1 sin t),  IL 2 = IL 1 rotated by pi about z: (x, y, z) -> (-x, -y, z),
with the tilt angle between the coil *planes* (not their normals):
  y_s1 = R_c sqrt((1 - cos tilt) / 2),  z_s1 = R_c sqrt((1 + cos tilt) / 2).
The orientation (z = -z_s1 sin t) gives positive rotational transform in VMEC's sign convention.
Circles are exact at Fourier order 1, so the file holds the coils without discretisation error.
"""
import json
import math
import os

TILT_DEG = 64.0           # angle between the planes of the two IL coils
R_C, X_C0 = 0.405, 0.313  # IL coil radius and centre offset, m
I_IL = 170e3              # A, each IL coil
R_PF, Z_PF = 1.08, 0.405  # PF ring radius and height, m
I_PF = 40.3e3             # A, each PF ring

# A flux surface inside the region of closed surfaces: rotational transform 0.2079427045422565, the noble
# number between the 1/5 and 2/9 island chains, about 0.69 of the minor radius of the last closed surface.
# Flux for mu0 = 4 pi 1e-7.
REQUESTED_FLUX_WB = 0.011624274070939846


def coil(xc, ys, zc, zs, current):
    return {"xc": xc, "xs": [0.0, 0.0], "yc": [0.0, 0.0], "ys": ys, "zc": zc, "zs": zs, "current_A": current}


def cnt_coils(tilt_deg=TILT_DEG):
    c = math.cos(math.radians(tilt_deg))
    y_s1 = R_C * math.sqrt((1 - c) / 2)
    z_s1 = R_C * math.sqrt((1 + c) / 2)
    return [
        coil([X_C0, R_C], [0.0, -y_s1], [0.0, 0.0], [0.0, -z_s1], I_IL),       # IL 1
        coil([-X_C0, -R_C], [0.0, y_s1], [0.0, 0.0], [0.0, -z_s1], I_IL),      # IL 2 = (-x, -y, z) of IL 1
        coil([0.0, R_PF], [0.0, R_PF], [Z_PF, 0.0], [0.0, 0.0], I_PF),         # PF top, counter-clockwise from +z
        coil([0.0, R_PF], [0.0, R_PF], [-Z_PF, 0.0], [0.0, 0.0], I_PF),        # PF bottom
    ]


if __name__ == "__main__":
    d = {"id": "cnt",
         "family": "classical (interlocked coils)",
         "source": "Columbia Non-neutral Torus, 64 deg tilt, I_IL = 170 kA, I_PF = 40.3 kA: Pedersen et al., "
                   "Fusion Sci. Technol. 46, 200 (2004); coil parameterisation of Yu et al., Phys. Plasmas 28, "
                   "092501 (2021); written by make_cnt.py",
         "nfp": 2, "stellarator_symmetric": True, "units": {"length": "m", "current": "A"},
         "coils": cnt_coils(),
         "requested_toroidal_flux_Wb": REQUESTED_FLUX_WB}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnt.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
        f.write("\n")
    print(f"wrote {path}")
