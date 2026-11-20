import sys
from pathlib import Path
import numpy as np

dat_path = Path(sys.argv[1])
data = np.loadtxt(dat_path, comments="#")

step, u_grip, d_if, d_cz, tn_avg, tn_top, tn_bot, T_mob = data.T

# 1 GPa * 1 A = 0.1 J/m^2
t_peak = np.max(tn_avg)
idx_peak = np.argmax(tn_avg)
d_peak_cz = d_cz[idx_peak]
d_peak_grip = u_grip[idx_peak]

# Trabalho de separacao integrado numericamente (apenas parte positiva ou curva completa)
G_grip = np.trapezoid(tn_avg, u_grip) * 0.1 if hasattr(np, "trapezoid") else np.trapz(tn_avg, u_grip) * 0.1
G_cz   = np.trapezoid(tn_avg, d_cz)   * 0.1 if hasattr(np, "trapezoid") else np.trapz(tn_avg, d_cz)   * 0.1
G_if   = np.trapezoid(tn_avg, d_if)   * 0.1 if hasattr(np, "trapezoid") else np.trapz(tn_avg, d_if)   * 0.1

imb_rms = np.sqrt(np.mean((tn_top - tn_bot) ** 2))

print(f"=== Analise Modo I: {dat_path.name} ===")
print(f"Tensao de pico t_n^0      = {t_peak:.3f} GPa (em d_cz = {d_peak_cz:.3f} A, u_grip = {d_peak_grip:.3f} A)")
print(f"Tensao final (separado)   = {tn_avg[-1]:.4f} GPa")
print(f"Trabalho G_Ic (d_cz)      = {G_cz:.3f} J/m^2")
print(f"Trabalho G_Ic (d_if)      = {G_if:.3f} J/m^2")
print(f"Trabalho G_Ic (u_grip)    = {G_grip:.3f} J/m^2")
print(f"Desequilibrio RMS top/bot = {imb_rms:.3f} GPa")
print(f"Temp media zona movel     = {np.mean(T_mob):.1f} +/- {np.std(T_mob):.1f} K")

try:
    import matplotlib.pyplot as plt
    out_fig = dat_path.parents[2] / "05_report_figs" / f"{dat_path.stem}.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(d_cz, tn_avg, lw=2, label=r"$T_n(\delta_{\mathrm{cz}})$")
    ax.plot(u_grip, tn_avg, "--", lw=1.3, alpha=0.8, label=r"$T_n(u_{\mathrm{grip}})$")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel(r"Separacao $\delta$ ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Tracao normal $T_n$ (GPa)")
    ax.set_title(f"Modo I Al/SiC — {dat_path.stem} ($t_n^0={t_peak:.2f}$ GPa, $G_{{Ic}}={G_cz:.2f}$ J/m$^2$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig)
    print(f"Grafico guardado em: {out_fig}")
except Exception as e:
    print(f"(Aviso matplotlib: {e})")
