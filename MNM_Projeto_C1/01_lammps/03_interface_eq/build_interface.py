import math
from pathlib import Path

def build_al_sic_bicrystal(
    nx_sic=6, ny_sic=6, nz_sic=8,
    nx_al=9, ny_al=9, nz_al=10,
    a_sic=4.32117, a_al=4.05000,
    gap=2.40,
    out_file="data.Al_SiC_init",
):
    # Dimensoes laterais medias para minimizar tensao inicial (misfit = 0.59%)
    lx_sic = nx_sic * a_sic
    lx_al = nx_al * (a_al / math.sqrt(2.0))
    Lx = 0.5 * (lx_sic + lx_al)
    Ly = Lx

    ax_sic, ay_sic, az_sic = Lx / nx_sic, Ly / ny_sic, a_sic
    ax_al, ay_al, az_al = Lx / nx_al, Ly / ny_al, a_al

    # Base zinc-blende 3C-SiC: C em z=0,0.5 e Si em z=0.25,0.75 -> topo e Si-terminado
    c_basis = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    si_basis = [(0.25, 0.25, 0.25), (0.25, 0.75, 0.75), (0.75, 0.25, 0.75), (0.75, 0.75, 0.25)]

    # Base FCC rodada 45 deg (celula de Bain: x=[110], y=[-110], z=[001])
    al_basis = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]

    atoms = []  # (type, x, y, z) -> 1: Al, 2: Si, 3: C

    # 1. Bloco inferior: 3C-SiC
    for ix in range(nx_sic):
        for iy in range(ny_sic):
            for iz in range(nz_sic):
                for bx, by, bz in c_basis:
                    atoms.append((3, (ix + bx) * ax_sic, (iy + by) * ay_sic, (iz + bz) * az_sic))
                for bx, by, bz in si_basis:
                    atoms.append((2, (ix + bx) * ax_sic, (iy + by) * ay_sic, (iz + bz) * az_sic))

    z_sic_top = max(a[3] for a in atoms)
    z_al_start = z_sic_top + gap

    # 2. Bloco superior: Al FCC
    for ix in range(nx_al):
        for iy in range(ny_al):
            for iz in range(nz_al):
                for bx, by, bz in al_basis:
                    atoms.append((1, (ix + bx) * ax_al, (iy + by) * ay_al, z_al_start + (iz + bz) * az_al))

    z_min = min(a[3] for a in atoms) - 10.0
    z_max = max(a[3] for a in atoms) + 10.0

    n_al = sum(1 for a in atoms if a[0] == 1)
    n_si = sum(1 for a in atoms if a[0] == 2)
    n_c = sum(1 for a in atoms if a[0] == 3)

    with open(out_file, "w") as f:
        f.write("Al(100)[110] / 3C-SiC(100)[100] Si-terminated bicrystal\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write("3 atom types\n\n")
        f.write(f"0.000000 {Lx:.6f} xlo xhi\n")
        f.write(f"0.000000 {Ly:.6f} ylo yhi\n")
        f.write(f"{z_min:.6f} {z_max:.6f} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 26.981539 # Al\n")
        f.write("2 28.085500 # Si\n")
        f.write("3 12.011000 # C\n\n")
        f.write("Atoms # atomic\n\n")
        for idx, (atype, x, y, z) in enumerate(atoms, start=1):
            f.write(f"{idx} {atype} {x:.6f} {y:.6f} {z:.6f}\n")

    misfit = 100.0 * abs(lx_sic - lx_al) / lx_sic
    print(f"Escrito {out_file}: {len(atoms)} atomos (Al={n_al}, Si={n_si}, C={n_c})")
    print(f"Lx = Ly = {Lx:.4f} A (misfit = {misfit:.2f}%)")
    print(f"SiC z: [0.00, {z_sic_top:.2f}] A | Al z: [{z_al_start:.2f}, {z_max-10.0:.2f}] A")

if __name__ == "__main__":
    build_al_sic_bicrystal()
