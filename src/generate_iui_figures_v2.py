from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('/mnt/data/IUI2027_figures_v2')
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.0,
    'axes.titlesize': 9.2,
    'axes.labelsize': 8.9,
    'xtick.labelsize': 7.8,
    'ytick.labelsize': 7.8,
    'legend.fontsize': 7.8,
    'svg.fonttype': 'path',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

BLACK = '#111111'
DARK = '#444444'
MID = '#777777'
LIGHT = '#BBBBBB'
VERY_LIGHT = '#EEEEEE'
WHITE = '#FFFFFF'


def save_all(fig, stem):
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight', pad_inches=0.03, facecolor='white')
    fig.savefig(OUT / f'{stem}.png', dpi=300, bbox_inches='tight', pad_inches=0.03, facecolor='white')
    fig.savefig(OUT / f'{stem}.tiff', dpi=450, bbox_inches='tight', pad_inches=0.03, facecolor='white', pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

# --------------------
# Figure 3: revised
# --------------------
conditions = ['Clean','10 Hz','50 px\nnoise','40% random\nmissing','40% burst\nmissing','Noise +\nrandom','Noise +\nburst']
soft = [0.723,0.723,0.700,0.719,0.670,0.702,0.679]
hard = [0.720,0.720,0.701,0.715,0.671,0.702,0.681]
fixed = [0.640,0.594,0.504,0.596,0.608,0.502,0.502]
native = [0.700,0.518,0.538,0.500,0.621,0.500,0.501]

with open(OUT/'Figure_3_data.csv','w',newline='',encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['condition','soft_aoi_roc_auc','hard_aoi_roc_auc','fixed_ivt_30hz_roc_auc','native_tuned_ivt_roc_auc'])
    w.writerows(zip(conditions, soft, hard, fixed, native))

fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.9), sharex=True)
x = np.arange(len(conditions))

ax = axes[0]
ax.plot(x, soft, marker='o', markersize=4.7, mfc=WHITE, mec=BLACK, linestyle='-', linewidth=1.3, color=BLACK, label='Soft-AOI')
ax.plot(x, hard, marker='s', markersize=4.5, mfc=WHITE, mec=DARK, linestyle='--', linewidth=1.2, color=DARK, label='Hard-AOI')
ax.plot(x, fixed, marker='^', markersize=4.8, mfc=DARK, mec=DARK, linestyle='-.', linewidth=1.2, color=MID, label='Fixed I-VT')
ax.axhline(0.5, color=LIGHT, linewidth=1.0, linestyle=':')
ax.text(6.05, 0.504, 'chance', color=DARK, fontsize=7.0, va='bottom', ha='right')
ax.set_ylim(0.47, 0.75)
ax.set_ylabel('ROC-AUC')
ax.set_title('(a) Matched 30-Hz comparison', loc='left', weight='bold')
ax.grid(axis='y', color=VERY_LIGHT, linewidth=0.8)
ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.02))

ax = axes[1]
ax.plot(x, native, marker='D', markersize=4.7, mfc=WHITE, mec=BLACK, linestyle='-', linewidth=1.25, color=BLACK, label='Native-rate tuned I-VT')
ax.axhline(0.5, color=LIGHT, linewidth=1.0, linestyle=':')
ax.text(6.05, 0.504, 'chance', color=DARK, fontsize=7.0, va='bottom', ha='right')
ax.set_ylim(0.47, 0.75)
ax.set_ylabel('ROC-AUC')
ax.set_title('(b) Native-rate tuned I-VT fairness sensitivity', loc='left', weight='bold')
ax.grid(axis='y', color=VERY_LIGHT, linewidth=0.8)
ax.legend(frameon=False, loc='upper right')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_xlabel('Controlled sensing condition')

fig.tight_layout(h_pad=1.05)
save_all(fig, 'Figure_3_Measurement_Degradation_v2')

# --------------------
# Figure 4: revised
# --------------------
methods = ['Frozen','Participant\ncalibration','QCN only','QCN-PC']
eng = [0.562330, 0.602420, 0.591300, 0.649726]
tur = [0.645134, 0.711851, 0.677354, 0.718801]

with open(OUT/'Figure_4_transfer_data.csv','w',newline='',encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['method','english_roc_auc','turkish_roc_auc'])
    w.writerows(zip(methods, eng, tur))

# Values previously used / discussed in the paper draft
contrast = [
    ('English', 0.047306, 0.021402, 0.074433),
    ('Turkish', 0.006950, -0.021554, 0.034555),
]
with open(OUT/'Figure_4_contrast_data.csv','w',newline='',encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['target','qcn_pc_minus_calibration_auc','ci_low','ci_high'])
    w.writerows(contrast)

fig = plt.figure(figsize=(6.3, 3.45))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.95], wspace=0.42)

# Panel a: categorical dot plot (no connecting lines)
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(methods))
offset = 0.09
ax.scatter(x - offset, eng, s=34, facecolors=WHITE, edgecolors=BLACK, marker='o', linewidths=1.2, label='English development', zorder=3)
ax.scatter(x + offset, tur, s=34, facecolors=BLACK, edgecolors=BLACK, marker='s', linewidths=0.9, label='Locked Turkish', zorder=3)
for xi, yi in zip(x - offset, eng):
    ax.plot([xi, xi], [0.53, yi], color=VERY_LIGHT, linewidth=0.6, zorder=1)
for xi, yi in zip(x + offset, tur):
    ax.plot([xi, xi], [0.53, yi], color=VERY_LIGHT, linewidth=0.6, zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0.53, 0.75)
ax.set_ylabel('ROC-AUC')
ax.set_title('(a) Frozen transfer and target-side adaptation', loc='left', weight='bold')
ax.grid(axis='y', color=VERY_LIGHT, linewidth=0.8)
ax.legend(frameon=False, loc='lower right')

# Panel b: forest plot
ax = fig.add_subplot(gs[0, 1])
y = np.array([1, 0])
vals = np.array([c[1] for c in contrast])
lo = np.array([c[2] for c in contrast])
hi = np.array([c[3] for c in contrast])
ax.errorbar(vals, y, xerr=np.vstack([vals - lo, hi - vals]), fmt='o', color=BLACK, ecolor=DARK, capsize=4, linewidth=1.1)
ax.axvline(0, color=LIGHT, linestyle=':', linewidth=1.0)
ax.set_yticks(y)
ax.set_yticklabels(['English', 'Turkish'])
ax.set_xlim(-0.035, 0.085)
ax.set_xlabel('ROC-AUC difference\n(QCN-PC − calibration)')
ax.set_title('(b) Incremental QCN contrast\n95% cluster CI', loc='left', weight='bold', fontsize=8.8)
ax.grid(axis='x', color=VERY_LIGHT, linewidth=0.8)

fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.17)
save_all(fig, 'Figure_4_Cross_Language_Calibration_v2')

# --------------------
# Figure 5: revised
# --------------------
# Panel (a): participant calibration only, participant-balanced coverage sensitivity at lambda=1.5
coverage_labels = ['50%', '20%', '10%']
coverage_values = np.array([50, 20, 10])
eng_save = np.array([9.6664008, 12.2170108, 10.6828959])
eng_lo = np.array([2.7257683, 1.7053667, -1.1909306])
eng_hi = np.array([16.4224931, 21.6754776, 21.1185403])
tur_save = np.array([10.6050371, 3.7754767, -6.7524639])
tur_lo = np.array([0.9363296, -11.77097395, -24.4379277])
tur_hi = np.array([19.0132210, 17.0231779, 8.3135392])

# Panel (b): strictly past-only + fixed English confidence threshold
future_free = [
    ('English', 50.0, 8.4026, 0.5041, 16.0212, 1.2658),
    ('Turkish', 45.5, 15.6093, 6.5062, 24.2504, 1.1493),
]

with open(OUT/'Figure_5_policy_cost_data.csv','w',newline='',encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['panel','target','coverage_pct','saving_pct','ci_low_pct','ci_high_pct','break_even_lambda'])
    for cov, s, lo, hi in zip(coverage_values, eng_save, eng_lo, eng_hi):
        w.writerow(['coverage_sensitivity','English',cov,s,lo,hi,''])
    for cov, s, lo, hi in zip(coverage_values, tur_save, tur_lo, tur_hi):
        w.writerow(['coverage_sensitivity','Turkish',cov,s,lo,hi,''])
    for target, cov, s, lo, hi, be in future_free:
        w.writerow(['future_free_policy',target,cov,s,lo,hi,be])

fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2), sharey=False)

# Panel a
ax = axes[0]
x = np.arange(len(coverage_labels))
ax.errorbar(x - 0.04, eng_save, yerr=np.vstack([eng_save - eng_lo, eng_hi - eng_save]), fmt='o-', markersize=4.8, mfc=WHITE, mec=BLACK, color=BLACK, linewidth=1.2, capsize=3, label='English')
ax.errorbar(x + 0.04, tur_save, yerr=np.vstack([tur_save - tur_lo, tur_hi - tur_save]), fmt='s--', markersize=4.6, mfc=BLACK, mec=BLACK, color=DARK, linewidth=1.1, capsize=3, label='Turkish')
ax.axhline(0, color=LIGHT, linestyle=':', linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels(coverage_labels)
ax.set_xlabel('Participant-balanced coverage')
ax.set_ylabel('Estimated relative cost saving\nat λ = 1.5 (%)')
ax.set_title('(a) Participant calibration only\ncoverage sensitivity', loc='left', weight='bold', fontsize=8.8)
ax.grid(axis='y', color=VERY_LIGHT, linewidth=0.8)
ax.legend(frameon=False, loc='upper right')
ax.set_ylim(-28, 28)

# Panel b
ax = axes[1]
x2 = np.arange(2)
savings = np.array([row[2] for row in future_free])
lo2 = np.array([row[3] for row in future_free])
hi2 = np.array([row[4] for row in future_free])
ax.errorbar(x2, savings, yerr=np.vstack([savings - lo2, hi2 - savings]), fmt='o', markersize=5.0, mfc=WHITE, mec=BLACK, color=BLACK, ecolor=DARK, linewidth=1.15, capsize=4)
ax.axhline(0, color=LIGHT, linestyle=':', linewidth=1.0)
ax.set_xticks(x2)
ax.set_xticklabels([f"{row[0]}\n({row[1]:.1f}% cov.)" for row in future_free])
ax.set_xlabel('Transferred future-free policy')
ax.set_title('(b) Past-only calibration + English-fixed margin', loc='left', weight='bold', fontsize=8.8)
ax.grid(axis='y', color=VERY_LIGHT, linewidth=0.8)
ax.set_ylim(-28, 28)
for xi, row, s in zip(x2, future_free, savings):
    target, coverage, saving, ci_low, ci_high, break_even = row
    ax.text(xi, -24.2, f"λ*={break_even:.3f}", ha='center', va='bottom', fontsize=7.0, color=DARK)

fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.22, wspace=0.32)
save_all(fig, 'Figure_5_Act_Abstain_Cost_v2')

print('Created revised figures in', OUT)
