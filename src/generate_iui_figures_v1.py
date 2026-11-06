from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = Path('/mnt/data/IUI2027_figures_v1')
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.8,
    'axes.titlesize': 9.4,
    'axes.labelsize': 8.9,
    'xtick.labelsize': 7.8,
    'ytick.labelsize': 7.8,
    'legend.fontsize': 7.8,
    'svg.fonttype': 'path',
    'pdf.fonttype': 42,
})

BLACK = '#111111'; DARK = '#444444'; MID = '#777777'; LIGHT = '#BBBBBB'; VERY_LIGHT = '#EEEEEE'; WHITE = '#FFFFFF'

def save_all(fig, stem):
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight', pad_inches=0.04)
    fig.savefig(OUT / f'{stem}.png', dpi=300, bbox_inches='tight', pad_inches=0.04, facecolor='white')
    fig.savefig(OUT / f'{stem}.tiff', dpi=450, bbox_inches='tight', pad_inches=0.04, facecolor='white', pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def box(ax, xy, wh, text, lw=1.0, fc=WHITE, ec=BLACK, fontsize=7.2, weight='normal', radius=0.018):
    x,y=xy; w,h=wh
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.008,rounding_size={radius}',linewidth=lw,edgecolor=ec,facecolor=fc)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,weight=weight,color=BLACK,linespacing=1.13)
    return p

def arrow(ax,start,end,lw=0.95,ls='-'):
    a=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,linewidth=lw,linestyle=ls,color=DARK,shrinkA=2,shrinkB=2)
    ax.add_patch(a); return a

# Figure 1
fig,ax=plt.subplots(figsize=(6.2,3.35)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.text(0.5,0.97,'Reliability layer: from reading goal to an act-or-abstain decision',ha='center',va='top',fontsize=9.8,weight='bold')
ax.text(0.5,0.91,'What is the reader trying to do?',ha='center',va='top',fontsize=7.6,color=DARK)
box(ax,(0.09,0.76),(0.34,0.11),'Ordinary reading (NR)\nUnderstand overall content',fc=VERY_LIGHT,fontsize=7.4,weight='bold')
box(ax,(0.57,0.76),(0.34,0.11),'Information seeking (IS/TSR)\nFind specific information',fc=VERY_LIGHT,fontsize=7.4,weight='bold')
box(ax,(0.39,0.625),(0.22,0.065),'Continuous gaze samples',fontsize=7.2,weight='bold')
arrow(ax,(0.26,0.76),(0.46,0.69)); arrow(ax,(0.74,0.76),(0.54,0.69))

stage_x=[0.035,0.275,0.515,0.755]
stages=[
    ('1  Event-free AOI\nMap samples to words\nNo event boundaries','ZuCo 2.0'),
    ('2  Robust source training\nPaired clean / degraded\nquality views','OneStop'),
    ('3  Self-calibration\nCenter frozen scores from\nunlabeled history','WebQAmGaze'),
    ('4  Confidence policy\nFinal margin:\nACT / ABSTAIN','WebQAmGaze'),
]
for i,(x,(text,ds)) in enumerate(zip(stage_x,stages)):
    box(ax,(x,0.35),(0.205,0.17),text,fontsize=6.15,weight='bold' if i in (0,3) else 'normal')
    ax.text(x+0.1025,0.318,ds,ha='center',va='top',fontsize=6.35,color=DARK)
    if i<3: arrow(ax,(x+0.205,0.435),(stage_x[i+1],0.435))
arrow(ax,(0.5,0.625),(0.137,0.52))
box(ax,(0.595,0.095),(0.17,0.07),'ACT\nHighlight / navigate',fc=VERY_LIGHT,fontsize=5.9,weight='bold')
box(ax,(0.79,0.095),(0.17,0.07),'ABSTAIN\nLeave unchanged / wait',fontsize=5.9,weight='bold')
arrow(ax,(0.857,0.35),(0.680,0.165)); arrow(ax,(0.857,0.35),(0.875,0.165))
ax.text(0.5,0.025,'The interface acts only when reading-goal evidence is sufficiently reliable.',ha='center',va='bottom',fontsize=6.8,color=DARK)
fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
save_all(fig,'Figure_1_Reliability_Layer')

# Figure 2
fig,ax=plt.subplots(figsize=(6.2,3.35)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.text(0.5,0.965,'Continuous gaze samples mapped directly to word AOIs',ha='center',va='top',fontsize=9.8,weight='bold')
rows=[(['Readers','search','for','relevant','information'],0.68),(['while','ordinary','reading','covers','the','whole','text'],0.43)]
rects=[]
for words,y in rows:
    x=0.04
    for w in words:
        width=max(0.07,0.0105*len(w)+0.040)
        r=Rectangle((x,y),width,0.10,facecolor=WHITE,edgecolor=MID,linewidth=0.9)
        ax.add_patch(r); ax.text(x+width/2,y+0.05,w,ha='center',va='center',fontsize=6.8)
        rects.append((w,x,y,width)); x+=width+0.010
pts=[(0.17,0.60),(0.34,0.56),(0.52,0.61),(0.67,0.59),(0.32,0.37),(0.49,0.39)]
for x,y in pts: ax.add_patch(Circle((x,y),0.008,facecolor=WHITE,edgecolor=BLACK,linewidth=1.0))
amb=(0.70,0.61); ax.add_patch(Circle(amb,0.010,facecolor=WHITE,edgecolor=BLACK,linewidth=1.15))
rel=[r for r in rects if r[0]=='relevant'][0]; inf=[r for r in rects if r[0]=='information'][0]
relc=(rel[1]+rel[3]/2,rel[2]+0.05); infc=(inf[1]+inf[3]/2,inf[2]+0.05)
arrow(ax,amb,relc,lw=0.75); arrow(ax,amb,infc,lw=0.75)
ax.text(0.64,0.64,'soft memberships',fontsize=6.7,color=DARK)
ax.text(0.68,0.335,r'$q_j(t)=\exp[-d(g_t,A_j)^2/(2\sigma^2)]$',fontsize=7.4,ha='left')
ax.text(0.68,0.275,r'$p_j(t)=q_j(t)/\sum_k q_k(t)$',fontsize=7.4,ha='left')
ax.text(0.68,0.215,'No fixation or saccade boundary is required.',fontsize=7.0,ha='left',weight='bold')
ax.text(0.68,0.17,'Trial summaries aggregate occupancy, progress,\ntransitions, and measurement quality.',fontsize=6.5,ha='left',va='top',color=DARK)
ax.text(0.04,0.255,'Boundary-handling control',fontsize=7.2,weight='bold')
box(ax,(0.04,0.075),(0.25,0.11),'Soft-AOI\nretain normalized memberships',fc=VERY_LIGHT,fontsize=6.6)
box(ax,(0.35,0.075),(0.25,0.11),'Hard-AOI\nretain highest-membership word',fontsize=6.6)
arrow(ax,(0.29,0.13),(0.35,0.13))
ax.text(0.04,0.025,'Same gaze samples and word rectangles; only the assignment rule changes.',fontsize=6.6,color=DARK)
fig.subplots_adjust(left=0.01,right=0.99,top=0.99,bottom=0.01)
save_all(fig,'Figure_2_Event_Free_AOI_Mapping')

# Figure 3
conditions=['Clean','10 Hz','50 px\nnoise','40% random\nmissing','40% burst\nmissing','Noise +\nrandom','Noise +\nburst']
soft=[0.723,0.723,0.700,0.719,0.670,0.702,0.679]; hard=[0.720,0.720,0.701,0.715,0.671,0.702,0.681]; fixed=[0.640,0.594,0.504,0.596,0.608,0.502,0.502]; native=[0.700,0.518,0.538,0.500,0.621,0.500,0.501]
with open(OUT/'Figure_3_data.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['condition','soft_aoi_roc_auc','hard_aoi_roc_auc','fixed_ivt_30hz_roc_auc','native_tuned_ivt_roc_auc']); w.writerows(zip(conditions,soft,hard,fixed,native))
fig,axes=plt.subplots(2,1,figsize=(6.2,4.65),sharex=True); x=np.arange(len(conditions))
ax=axes[0]; ax.plot(x,soft,marker='o',linestyle='-',color=BLACK,label='Soft-AOI'); ax.plot(x,hard,marker='s',linestyle='--',color=MID,label='Hard-AOI'); ax.plot(x,fixed,marker='^',linestyle='-.',color=DARK,label='Fixed I-VT'); ax.axhline(0.5,color=LIGHT,linewidth=1,linestyle=':'); ax.set_ylim(0.47,0.75); ax.set_ylabel('ROC-AUC'); ax.set_title('(a) Matched 30-Hz input',loc='left',weight='bold'); ax.grid(axis='y',color=VERY_LIGHT,linewidth=0.8); ax.legend(ncol=3,frameon=False,loc='lower left')
ax=axes[1]; ax.plot(x,native,marker='D',linestyle='-',color=BLACK,label='Native-rate tuned I-VT'); ax.axhline(0.5,color=LIGHT,linewidth=1,linestyle=':'); ax.set_ylim(0.47,0.75); ax.set_ylabel('ROC-AUC'); ax.set_title('(b) Native-rate fairness sensitivity',loc='left',weight='bold'); ax.grid(axis='y',color=VERY_LIGHT,linewidth=0.8); ax.legend(frameon=False,loc='upper right'); ax.set_xticks(x); ax.set_xticklabels(conditions); ax.set_xlabel('Controlled sensing condition')
fig.tight_layout(h_pad=1.0); save_all(fig,'Figure_3_Measurement_Degradation')

# Figure 4
methods=['Frozen','Participant\ncalibration','QCN only','QCN-PC']; eng=[0.5623,0.6024,0.5913,0.6497]; tur=[0.6451,0.7119,0.6774,0.7190]
with open(OUT/'Figure_4_transfer_data.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['method','english_roc_auc','turkish_roc_auc']); w.writerows(zip(methods,eng,tur))
contrast=[('English',0.047306,0.021402,0.074433),('Turkish',0.006950,-0.021554,0.034555)]
with open(OUT/'Figure_4_contrast_data.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['target','qcn_pc_minus_calibration_auc','ci_low','ci_high']); w.writerows(contrast)
fig=plt.figure(figsize=(6.2,3.35)); gs=fig.add_gridspec(1,2,width_ratios=[1.65,0.95],wspace=0.42)
ax=fig.add_subplot(gs[0,0]); x=np.arange(len(methods)); ax.plot(x,eng,marker='o',linestyle='-',color=BLACK,label='English development'); ax.plot(x,tur,marker='s',linestyle='--',color=MID,label='Locked Turkish'); ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylim(0.53,0.75); ax.set_ylabel('ROC-AUC'); ax.set_title('(a) Frozen transfer and adaptation',loc='left',weight='bold'); ax.grid(axis='y',color=VERY_LIGHT,linewidth=0.8); ax.legend(frameon=False,loc='lower right')
ax=fig.add_subplot(gs[0,1]); y=np.array([1,0]); vals=np.array([c[1] for c in contrast]); lo=np.array([c[2] for c in contrast]); hi=np.array([c[3] for c in contrast]); ax.errorbar(vals,y,xerr=np.vstack([vals-lo,hi-vals]),fmt='o',color=BLACK,ecolor=DARK,capsize=4,linewidth=1.05); ax.axvline(0,color=LIGHT,linestyle=':',linewidth=1); ax.set_yticks(y); ax.set_yticklabels(['English','Turkish']); ax.set_xlim(-0.035,0.085); ax.set_xlabel('ROC-AUC difference\n(QCN-PC − calibration)'); ax.set_title('(b) Incremental QCN contrast\n95% cluster CI',loc='left',weight='bold',fontsize=8.8); ax.grid(axis='x',color=VERY_LIGHT,linewidth=0.8)
fig.subplots_adjust(left=0.08,right=0.99,top=0.91,bottom=0.17,wspace=0.42); save_all(fig,'Figure_4_Cross_Language_Calibration')

# Figure 5
policy_data=[('Retrospective participant-balanced 50%','English',50.0,9.7,2.7,16.4,1.224),('Retrospective participant-balanced 50%','Turkish',50.0,10.6,0.9,19.0,1.234),('Past-only + English-fixed margin','English',50.0,8.4,0.5,16.0,1.266),('Past-only + English-fixed margin','Turkish',45.5,15.6,6.5,24.3,1.149)]
with open(OUT/'Figure_5_policy_cost_data.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['policy','target','realized_coverage_pct','saving_at_lambda_1_5_pct','ci_low_pct','ci_high_pct','break_even_lambda']); w.writerows(policy_data)
fig,axes=plt.subplots(1,2,figsize=(6.2,3.0),sharey=True)
configs=[(axes[0],'Retrospective participant-balanced 50%','(a) Participant calibration\nretrospective 50% selection'),(axes[1],'Past-only + English-fixed margin','(b) Past-only calibration\nEnglish-fixed margin')]
for ax,policy,title in configs:
    rows=[r for r in policy_data if r[0]==policy]; xs=np.arange(2); vals=np.array([r[3] for r in rows]); lo=np.array([r[4] for r in rows]); hi=np.array([r[5] for r in rows]); ax.errorbar(xs,vals,yerr=np.vstack([vals-lo,hi-vals]),fmt='o',markersize=5.5,color=BLACK,ecolor=DARK,capsize=4,linewidth=1.1); ax.axhline(0,color=LIGHT,linestyle=':',linewidth=1); ax.set_xticks(xs); ax.set_xticklabels([r[1] for r in rows]); ax.set_title(title,loc='left',weight='bold',fontsize=8.8); ax.grid(axis='y',color=VERY_LIGHT,linewidth=0.8); note='\n'.join([f"{r[1]}: coverage {r[2]:.1f}%, λ*={r[6]:.3f}" for r in rows]); ax.text(0.03,0.97,note,transform=ax.transAxes,ha='left',va='top',fontsize=6.6,color=DARK)
axes[0].set_ylabel('Estimated relative cost saving\nat λ = 1.5 (%)'); axes[0].set_ylim(-3,28)
fig.subplots_adjust(left=0.10,right=0.99,top=0.86,bottom=0.18,wspace=0.32); save_all(fig,'Figure_5_Act_Abstain_Cost')

print('Created/refined figures in',OUT)
