import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

trapz = getattr(np, "trapezoid", np.trapz)

PROJ = Path("/Users/David/Desktop/UNI/Mestrado/2 ano/Micro e Nano Mecânica/MNM_Projeto_C1")
M1_DIR = PROJ / "01_lammps" / "04_modeI_rate_sweep"
M2_DIR = PROJ / "01_lammps" / "05_modeII_shear"
FIG_DIR = PROJ / "05_report_figs"
OUT_JSON = PROJ / "02_python_bridge" / "czm_params_abaqus.json"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def moving_avg(y, win=5):
    if win <= 1:
        return y.copy()
    pad = win // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(yp, kernel, mode="valid")[:len(y)]


def load_dat(path):
    return np.loadtxt(path, comments="#")


def analyze_modeI(path, smooth_win=5):
    d = load_dat(path)
    # Cols: TimeStep u_grip d_if d_cz tn_avg tn_top tn_bot T_mob
    u_grip, d_if, d_cz = d[:, 1], d[:, 2], d[:, 3]
    tn_avg, tn_top, tn_bot, T_mob = d[:, 4], d[:, 5], d[:, 6], d[:, 7]
    tn_s = moving_avg(tn_avg, smooth_win)

    idx_raw = int(np.argmax(tn_avg))
    idx_s = int(np.argmax(tn_s))

    # Primeiro cruzamento de zero apos o pico (evita integrar ringing pos-rotura)
    post = np.where((np.arange(len(tn_s)) > idx_s) & (tn_s <= 0.0))[0]
    idx_zero = int(post[0]) if len(post) > 0 else len(tn_s) - 1

    # 1 GPa * 1 A = 0.1 J/m^2
    G_to_zero = float(trapz(tn_avg[:idx_zero + 1], d_cz[:idx_zero + 1]) * 0.1)
    G_total = float(trapz(tn_avg, d_cz) * 0.1)
    G_pos = float(trapz(np.maximum(tn_avg, 0.0), d_cz) * 0.1)

    # Rigidez elastica inicial Kn (fit linear entre 15% e 60% do pico)
    t_peak_s = float(tn_s[idx_s])
    mask_el = (np.arange(len(tn_s)) < idx_s) & (tn_s >= 0.15 * t_peak_s) & (tn_s <= 0.60 * t_peak_s)
    if np.sum(mask_el) >= 3:
        Kn_GPa_A = float(np.polyfit(d_cz[mask_el], tn_s[mask_el], 1)[0])
    else:
        Kn_GPa_A = t_peak_s / max(float(d_cz[idx_s]), 1e-6)

    # Parametros bilineares equivalentes (conservando tn0 e GIc)
    df_bilin_A = 2.0 * (G_to_zero / 0.1) / t_peak_s  # em Angstrom
    rms_imb = float(np.sqrt(np.mean((tn_top - tn_bot) ** 2)))

    return {
        "name": path.stem,
        "u_grip": u_grip, "d_if": d_if, "d_cz": d_cz,
        "tn_avg": tn_avg, "tn_s": tn_s, "T_mob": T_mob,
        "t_peak_raw_GPa": float(tn_avg[idx_raw]),
        "t_peak_smooth_GPa": t_peak_s,
        "d_cz_peak_A": float(d_cz[idx_s]),
        "d_if_max_A": float(np.max(d_if)),
        "d_if_end_A": float(d_if[-1]),
        "d_cz_zero_A": float(d_cz[idx_zero]),
        "tn_end_GPa": float(tn_s[-1]),
        "G_Ic_to_zero_Jm2": G_to_zero,
        "G_Ic_total_Jm2": G_total,
        "G_Ic_pos_Jm2": G_pos,
        "Kn_GPa_per_A": Kn_GPa_A,
        "Kn_N_per_m3": Kn_GPa_A * 1.0e19,
        "df_bilin_A": df_bilin_A,
        "rms_imbalance_GPa": rms_imb,
        "T_mean_K": float(np.mean(T_mob)),
        "T_std_K": float(np.std(T_mob)),
    }


def analyze_modeII(path, smooth_win=9):
    d = load_dat(path)
    # Cols: TimeStep u_grip d_cz_s ts_avg ts_top ts_bot tn_avg T_mob
    u_grip, d_s = d[:, 1], d[:, 2]
    ts_avg, ts_top, ts_bot, tn_avg, T_mob = d[:, 3], d[:, 4], d[:, 5], d[:, 6], d[:, 7]
    ts_s = moving_avg(ts_avg, smooth_win)

    # 1o ciclo de escorregamento: ate ao 1o minimo/cruzamento apos 1.5 A
    mask_c1 = d_s <= 3.3
    idx_p1 = int(np.argmax(ts_s[mask_c1]))
    post_p1 = np.where((np.arange(len(ts_s)) > idx_p1) & (ts_s <= 0.15 * ts_s[idx_p1]))[0]
    idx_end1 = int(post_p1[0]) if len(post_p1) > 0 else int(np.argmin(np.abs(d_s - 3.0)))

    G_II_c1 = float(trapz(np.maximum(ts_avg[:idx_end1 + 1], 0.0), d_s[:idx_end1 + 1]) * 0.1)
    idx_glob = int(np.argmax(ts_s))

    # Rigidez tangencial inicial (0 a 1o pico)
    mask_el = (np.arange(len(ts_s)) < idx_p1) & (d_s >= 0.05) & (ts_s >= 0.15 * ts_s[idx_p1]) & (ts_s <= 0.70 * ts_s[idx_p1])
    Ks_GPa_A = float(np.polyfit(d_s[mask_el], ts_s[mask_el], 1)[0]) if np.sum(mask_el) >= 3 else float(ts_s[idx_p1] / max(d_s[idx_p1], 0.1))

    return {
        "name": path.stem,
        "u_grip": u_grip, "d_s": d_s, "ts_avg": ts_avg, "ts_s": ts_s, "tn_avg": tn_avg,
        "ts_peak1_raw_GPa": float(np.max(ts_avg[:idx_end1 + 1])),
        "ts_peak1_smooth_GPa": float(ts_s[idx_p1]),
        "d_s_peak1_A": float(d_s[idx_p1]),
        "d_s_end1_A": float(d_s[idx_end1]),
        "ts_global_raw_GPa": float(np.max(ts_avg)),
        "ts_global_smooth_GPa": float(ts_s[idx_glob]),
        "d_s_global_A": float(d_s[idx_glob]),
        "G_IIc_cycle1_Jm2": G_II_c1,
        "Ks_GPa_per_A": Ks_GPa_A,
        "Ks_N_per_m3": Ks_GPa_A * 1.0e19,
        "tn_mean_GPa": float(np.mean(tn_avg)),
        "rms_imbalance_GPa": float(np.sqrt(np.mean((ts_top - ts_bot) ** 2))),
    }


def check_fracture_plane_from_dump(dump_path):
    """Le o ultimo frame de um dump LAMMPS e conta quantos atomos de Al ficaram colados ao SiC."""
    if not dump_path.exists():
        return None
    lines = dump_path.read_text().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("ITEM: TIMESTEP")]
    if not starts:
        return None
    last = lines[starts[-1]:]
    atom_idx = [i for i, l in enumerate(last) if l.startswith("ITEM: ATOMS")][0]
    data = np.array([[float(x) for x in r.split()[:5]] for r in last[atom_idx + 1:] if r.strip()])
    types, zs = data[:, 1].astype(int), data[:, 4]

    z_sic_max = float(np.max(zs[types != 1]))
    z_al = np.sort(zs[types == 1])
    gaps = np.diff(z_al)
    max_gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_idx])
    z_crack_lo = float(z_al[max_gap_idx])
    z_crack_hi = float(z_al[max_gap_idx + 1])
    n_al_adhered = int(np.sum(z_al <= z_crack_lo + 1e-6))
    n_al_total = int(len(z_al))

    return {
        "dump_file": dump_path.name,
        "z_SiC_top_A": z_sic_max,
        "crack_gap_A": max_gap,
        "crack_z_range_A": [z_crack_lo, z_crack_hi],
        "Al_atoms_adhered_to_SiC": n_al_adhered,
        "Al_atoms_total": n_al_total,
        "frac_Al_adhered_pct": 100.0 * n_al_adhered / n_al_total,
    }


# 1. Analisar todos os ensaios
m1_files = [
    ("v1p0", M1_DIR / "modeI_v1p0.dat", 5),
    ("v1p0_dt0p5", M1_DIR / "modeI_v1p0_dt0p5.dat", 5),
    ("v0p25", M1_DIR / "modeI_v0p25.dat", 9),
    ("v0p10", M1_DIR / "modeI_v0p10.dat", 9),
]
m1 = {k: analyze_modeI(p, w) for k, p, w in m1_files}
m2 = analyze_modeII(M2_DIR / "modeII_v0p25.dat", smooth_win=9)
crack_info = check_fracture_plane_from_dump(M1_DIR / "dump.modeI_v0p25.lammpstrj")

# 2. Imprimir tabela resumo
print("\n=== RESUMO MODO I (Tracao Normal) ===")
hdr = f"{'Caso':<12} {'tn0_raw':>8} {'tn0_sm':>8} {'d_pk':>6} {'GIc_0':>7} {'GIc_tot':>8} {'Kn(GPa/A)':>10} {'df_bil':>7} {'d_if_end':>9} {'RMS':>6}"
print(hdr)
print("-" * len(hdr))
for k, r in m1.items():
    print(f"{k:<12} {r['t_peak_raw_GPa']:8.3f} {r['t_peak_smooth_GPa']:8.3f} {r['d_cz_peak_A']:6.2f} "
          f"{r['G_Ic_to_zero_Jm2']:7.3f} {r['G_Ic_total_Jm2']:8.3f} {r['Kn_GPa_per_A']:10.2f} "
          f"{r['df_bilin_A']:7.2f} {r['d_if_end_A']:9.3f} {r['rms_imbalance_GPa']:6.2f}")

print("\n=== RESUMO MODO II (Corte v=0.25 A/ps) ===")
print(f"1o ciclo: ts0_raw = {m2['ts_peak1_raw_GPa']:.3f} GPa | ts0_smooth = {m2['ts_peak1_smooth_GPa']:.3f} GPa (em d_s = {m2['d_s_peak1_A']:.2f} A)")
print(f"Pico global (2o ciclo): ts_raw = {m2['ts_global_raw_GPa']:.3f} GPa | ts_smooth = {m2['ts_global_smooth_GPa']:.3f} GPa (em d_s = {m2['d_s_global_A']:.2f} A)")
print(f"Trabalho 1o ciclo (ate {m2['d_s_end1_A']:.2f} A): G_IIc_c1 = {m2['G_IIc_cycle1_Jm2']:.3f} J/m^2 | Ks = {m2['Ks_GPa_per_A']:.2f} GPa/A")
print(f"Tensao normal media durante corte confinado: <Tn> = {m2['tn_mean_GPa']:.3f} GPa")

if crack_info:
    print("\n=== VERIFICACAO ATOMISTICA DO PLANO DE FRATURA (ultimo frame) ===")
    print(f"Ficheiro: {crack_info['dump_file']}")
    print(f"Topo SiC: z = {crack_info['z_SiC_top_A']:.2f} A")
    print(f"Maior lacuna (fenda) em Al: delta_z = {crack_info['crack_gap_A']:.2f} A entre z = {crack_info['crack_z_range_A'][0]:.2f} e {crack_info['crack_z_range_A'][1]:.2f} A")
    print(f"Atomos de Al aderidos ao SiC: {crack_info['Al_atoms_adhered_to_SiC']} / {crack_info['Al_atoms_total']} ({crack_info['frac_Al_adhered_pct']:.1f}%)")

# 3. Exportar JSON de parametros recomendados para o Abaqus
ref_I = m1["v0p25"]
ref_I_slow = m1["v0p10"]
czm_export = {
    "notes": "Unidades SI e mm-MPa-N (Abaqus consistente: mm, N, MPa, mJ/mm^2 = J/m^2 * 1e-3).",
    "modeI_v0p25_complete_separation": {
        "tn0_GPa": ref_I["t_peak_smooth_GPa"],
        "tn0_MPa": ref_I["t_peak_smooth_GPa"] * 1e3,
        "GIc_Jm2": ref_I["G_Ic_to_zero_Jm2"],
        "GIc_N_per_mm": ref_I["G_Ic_to_zero_Jm2"] * 1e-3,
        "Kn_GPa_per_A": ref_I["Kn_GPa_per_A"],
        "Kn_MPa_per_mm": ref_I["Kn_GPa_per_A"] * 1e10,
        "df_bilin_A": ref_I["df_bilin_A"],
    },
    "modeI_v0p10_quasistatic_limit": {
        "tn0_GPa": ref_I_slow["t_peak_smooth_GPa"],
        "tn0_MPa": ref_I_slow["t_peak_smooth_GPa"] * 1e3,
        "GIc_partial_to_8A_Jm2": ref_I_slow["G_Ic_total_Jm2"],
        "Kn_GPa_per_A": ref_I_slow["Kn_GPa_per_A"],
        "tn_residual_at_8A_GPa": ref_I_slow["tn_end_GPa"],
    },
    "modeII_v0p25_first_slip_cycle": {
        "ts0_cycle1_GPa": m2["ts_peak1_smooth_GPa"],
        "ts0_cycle1_MPa": m2["ts_peak1_smooth_GPa"] * 1e3,
        "ts_global_GPa": m2["ts_global_smooth_GPa"],
        "GIIc_cycle1_Jm2": m2["G_IIc_cycle1_Jm2"],
        "GIIc_cycle1_N_per_mm": m2["G_IIc_cycle1_Jm2"] * 1e-3,
        "Ks_GPa_per_A": m2["Ks_GPa_per_A"],
    },
    "fracture_plane_verification": crack_info,
}
OUT_JSON.write_text(json.dumps(czm_export, indent=2))
print(f"\nParametros CZM guardados em: {OUT_JSON}")

# 4. Figuras para o relatorio
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# Fig 1: Convergencia dt (v=1.0)
fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=180)
ax.plot(m1["v1p0"]["d_cz"], m1["v1p0"]["tn_s"], lw=1.8, label=r"$\Delta t = 1.0\ \mathrm{fs}$")
ax.plot(m1["v1p0_dt0p5"]["d_cz"], m1["v1p0_dt0p5"]["tn_s"], "--", lw=1.8, label=r"$\Delta t = 0.5\ \mathrm{fs}$")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"Separação da zona coesiva $\delta_{\mathrm{cz}}$ ($\mathrm{\AA}$)")
ax.set_ylabel(r"Tração normal $T_n$ (GPa)")
ax.set_title(r"Verificação de convergência temporal ($v = 1.0\ \mathrm{\AA/ps}$, 300 K)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_modeI_dt_convergence.png")
plt.close(fig)

# Fig 2: Rate sweep + prova d_if vs d_cz
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=180)
for k, lbl in [("v1p0", r"$1.00\ \mathrm{\AA/ps}$"), ("v0p25", r"$0.25\ \mathrm{\AA/ps}$"), ("v0p10", r"$0.10\ \mathrm{\AA/ps}$")]:
    ax1.plot(m1[k]["d_cz"], m1[k]["tn_s"], lw=1.8, label=lbl)
    ax2.plot(m1[k]["u_grip"], m1[k]["d_cz"], lw=1.8, label=rf"$\delta_{{\mathrm{{cz}}}}$ ({lbl})")
    ax2.plot(m1[k]["u_grip"], m1[k]["d_if"], "--", lw=1.3, alpha=0.8)
ax1.axhline(0, color="k", lw=0.7, ls=":")
ax1.set_xlabel(r"Separação $\delta_{\mathrm{cz}}$ ($\mathrm{\AA}$)")
ax1.set_ylabel(r"Tração normal $T_n$ (GPa)")
ax1.set_title("Modo I: Estudo de velocidade (Rate Sweep)")
ax1.legend()
ax2.set_xlabel(r"Deslocamento imposto $u_{\mathrm{grip}}$ ($\mathrm{\AA}$)")
ax2.set_ylabel(r"Separação medida ($\mathrm{\AA}$)")
ax2.set_title(r"$\delta_{\mathrm{cz}}$ (contínuo) vs. $\delta_{\mathrm{if}}$ (tracejado): falha no Al")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_modeI_rate_sweep_and_plane.png")
plt.close(fig)

# Fig 3: Modo II stick-slip
fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=180)
ax.plot(m2["d_s"], m2["ts_avg"], color="C0", alpha=0.25, lw=1.0, label="Bruto")
ax.plot(m2["d_s"], m2["ts_s"], color="C0", lw=2.0, label=r"$T_s(\delta_s)$ suavizado")
ax.plot(m2["d_s"], moving_avg(m2["tn_avg"], 9), color="C3", ls="--", lw=1.3, label=r"$T_n$ (confinamento $u_z=0$)")
ax.axvline(m2["d_s_end1_A"], color="gray", ls=":", lw=1.2, label=r"Fim do 1.º ciclo de corte")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlabel(r"Deslizamento tangencial $\delta_s$ ($\mathrm{\AA}$)")
ax.set_ylabel(r"Tensão (GPa)")
ax.set_title(r"Modo II Al/SiC ($v = 0.25\ \mathrm{\AA/ps}$): resposta periódica stick-slip")
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_modeII_shear.png")
plt.close(fig)

# Fig 4: Ajuste Bilinear CZM sobre MD (v0p25 e v0p10)
fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=180)
ax.plot(m1["v0p25"]["d_cz"], m1["v0p25"]["tn_s"], lw=1.8, label=r"MD $v=0.25\ \mathrm{\AA/ps}$")
ax.plot(m1["v0p10"]["d_cz"], m1["v0p10"]["tn_s"], lw=1.8, label=r"MD $v=0.10\ \mathrm{\AA/ps}$")

# Lei bilinear equivalente a v0p25
tn0 = ref_I["t_peak_smooth_GPa"]
d0 = tn0 / ref_I["Kn_GPa_per_A"]
df = ref_I["df_bilin_A"]
ax.plot([0.0, d0, df], [0.0, tn0, 0.0], "k--", lw=2.2,
        label=rf"CZM Bilinear ($t_n^0={tn0:.2f}$ GPa, $G_{{Ic}}={ref_I['G_Ic_to_zero_Jm2']:.2f}$ J/m$^2$)")
ax.axhline(0, color="k", lw=0.7, ls=":")
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 8.5)
ax.set_xlabel(r"Separação normal $\delta_n$ ($\mathrm{\AA}$)")
ax.set_ylabel(r"Tração normal $T_n$ (GPa)")
ax.set_title("Ponte MD $\\rightarrow$ Lei Coesiva Bilinear (Abaqus)")
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_czm_bilinear_fit.png")
plt.close(fig)

print(f"4 figuras guardadas em: {FIG_DIR}")
