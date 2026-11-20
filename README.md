# Micro-e-Nano-Mecanica

MNM_Projeto_C1_Cu/
├── 00_potentials/          # Cu_mishin1.eam.alloy
├── 01_lammps_bulk/         # Passo 1.1: a0, Ecoh, C_IJ (0 K e 300 K), Ev^f, NVE drift
├── 02_lammps_gb_310/       # Passo 1.2: gamma_s(310) + grelha 10x10 para gamma_GB(310)
├── 03_lammps_czm_pull/     # Passo 1.3: extração da lei coesiva (0 K passos 0.02 A + 300 K)
├── 04_lammps_bars/         # Passo 1.4: barras bicristal L = 5, 10, 20, 40 nm (0 K e 300 K)
├── 05_python_bridge/       # Rotação de C_IJ, subtração elástica, ajuste Bilinear + Xu-Needleman
├── 06_abaqus_bars/         # Scripts Python/.inp Abaqus 2023 (anisotrópico + COH3D8, L = 5..40 nm)
├── 07_report_figs/         # Figuras e tabelas finais + ficheiro commands_runtimes.txt
