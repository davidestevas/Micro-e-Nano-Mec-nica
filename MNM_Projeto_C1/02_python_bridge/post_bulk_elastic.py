import csv
import sys
from pathlib import Path

def analyze_cubic(csv_path, name, ref=None):
    rows = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["state"]] = {k: float(v) for k, v in r.items() if k != "state"}

    a0 = rows["eq"]["a0_A"]
    eps = rows["exx_pos"]["eps"] - rows["exx_neg"]["eps"]

    C11 = (rows["exx_pos"]["sxx_GPa"] - rows["exx_neg"]["sxx_GPa"]) / eps
    C12 = (rows["exx_pos"]["syy_GPa"] - rows["exx_neg"]["syy_GPa"]) / eps
    C13 = (rows["exx_pos"]["szz_GPa"] - rows["exx_neg"]["szz_GPa"]) / eps
    C44 = (rows["gxy_pos"]["sxy_GPa"] - rows["gxy_neg"]["sxy_GPa"]) / eps

    A_zener = 2.0 * C44 / (C11 - C12)
    born_ok = (C11 - C12 > 0) and (C44 > 0) and (C11 + 2 * C12 > 0)

    K = (C11 + 2.0 * C12) / 3.0
    G_V = (C11 - C12 + 3.0 * C44) / 5.0
    G_R = 5.0 * (C11 - C12) * C44 / (4.0 * C44 + 3.0 * (C11 - C12))
    G_H = 0.5 * (G_V + G_R)
    E_H = 9.0 * K * G_H / (3.0 * K + G_H)
    nu_H = (3.0 * K - 2.0 * G_H) / (2.0 * (3.0 * K + G_H))

    print(f"=== {name} (0 K) ===")
    print(f"a0   = {a0:.5f} A")
    print(f"C11  = {C11:.2f} GPa")
    print(f"C12  = {C12:.2f} GPa   (check C13 = {C13:.2f} GPa)")
    print(f"C44  = {C44:.2f} GPa")
    print(f"A    = {A_zener:.3f}   (Zener anisotropy)")
    print(f"Born = {'OK' if born_ok else 'FAIL'}")
    print(f"VRH  -> K = {K:.2f} GPa | G = {G_H:.2f} GPa | E = {E_H:.2f} GPa | nu = {nu_H:.4f}")
    if ref:
        print("Comparacao com literatura:")
        for k, val in [("a0", a0), ("C11", C11), ("C12", C12), ("C44", C44)]:
            err = 100.0 * (val - ref[k]) / ref[k]
            print(f"  {k:3s}: calc={val:.3f} vs ref={ref[k]:.3f} (erro = {err:+.2f}%)")

if __name__ == "__main__":
    p = Path(sys.argv[1])
    mat = sys.argv[2] if len(sys.argv) > 2 else "Material"
    refs = {
        "Al": {"a0": 4.050, "C11": 114.0, "C12": 61.6, "C44": 31.6},
        "SiC": {"a0": 4.3596, "C11": 390.0, "C12": 142.0, "C44": 256.0},
    }
    analyze_cubic(p, mat, refs.get(mat))
