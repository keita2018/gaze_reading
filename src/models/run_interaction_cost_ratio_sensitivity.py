#!/usr/bin/env python
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY = "qcn_pc"
REFERENCE = "participant_calibration"
EN_MAP = {
    "participant_only_oof": REFERENCE,
    "quality_removed_plus_participant": PRIMARY,
}
TR_COLS = {
    REFERENCE: "participant_median_loo_score",
    PRIMARY: "qcn_pc_score",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--english-predictions", type=Path,
                   default=Path("outputs/linear_feature_group_factorization_v1/target_feature_group_predictions.csv"))
    p.add_argument("--turkish-predictions", type=Path,
                   default=Path("outputs/iui_p0_confirmatory_TR_evaluation_v1/confirmatory_joined_trial_predictions.parquet"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/interaction_cost_ratio_sensitivity_v1"))
    p.add_argument("--coverages", default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1")
    p.add_argument("--key-lambdas", default="0.5,1,1.5,2,3,5,10,20")
    p.add_argument("--lambda-min", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=30.0)
    p.add_argument("--lambda-grid-points", type=int, default=181)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--progress-every", type=int, default=250)
    return p.parse_args()


def flist(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def base_cols(df):
    need = {"participant_id", "external_trial_id", "label_nr"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")
    df = df.copy()
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["external_trial_id"] = df["external_trial_id"].astype(str).str.strip()
    df["label_nr"] = pd.to_numeric(df["label_nr"], errors="raise").astype(int)
    if not df["label_nr"].isin([0, 1]).all():
        raise ValueError("label_nr must be 0/1")
    return df


def load_data(en_path: Path, tr_path: Path):
    if not en_path.is_file():
        raise FileNotFoundError(en_path)
    if not tr_path.is_file():
        raise FileNotFoundError(tr_path)

    en = base_cols(pd.read_csv(en_path, low_memory=False))
    for c in ["method", "score", "frozen_threshold"]:
        if c not in en.columns:
            raise ValueError(f"English table lacks {c}")
    en = en[en["method"].astype(str).isin(EN_MAP)].copy()
    en["method"] = en["method"].astype(str).map(EN_MAP)
    en["score"] = pd.to_numeric(en["score"], errors="raise").astype(float)
    en["frozen_threshold"] = pd.to_numeric(en["frozen_threshold"], errors="raise").astype(float)
    th = np.unique(np.round(en["frozen_threshold"].to_numpy(float), 12))
    if len(th) != 1:
        raise RuntimeError(f"Expected one frozen threshold, found {th.tolist()}")
    threshold = float(en["frozen_threshold"].iloc[0])
    en["dataset"] = "English"
    en["prediction"] = (en["score"] >= threshold).astype(int)
    en["confidence"] = np.abs(en["score"] - threshold)
    en["correct"] = (en["prediction"] == en["label_nr"]).astype(np.int8)
    en = en[["dataset","participant_id","external_trial_id","label_nr","method","score","prediction","confidence","correct"]]

    tr0 = base_cols(pd.read_parquet(tr_path))
    miss = set(TR_COLS.values()) - set(tr0.columns)
    if miss:
        raise ValueError(f"Turkish table lacks {sorted(miss)}")
    parts = []
    for method, col in TR_COLS.items():
        t = tr0[["participant_id","external_trial_id","label_nr"]].copy()
        t["dataset"] = "Turkish"
        t["method"] = method
        t["score"] = pd.to_numeric(tr0[col], errors="raise").astype(float)
        t["prediction"] = (t["score"] >= threshold).astype(int)
        t["confidence"] = np.abs(t["score"] - threshold)
        t["correct"] = (t["prediction"] == t["label_nr"]).astype(np.int8)
        parts.append(t)
    tr = pd.concat(parts, ignore_index=True)
    tr = tr[["dataset","participant_id","external_trial_id","label_nr","method","score","prediction","confidence","correct"]]
    return pd.concat([en, tr], ignore_index=True), threshold


def select_global(df, coverage):
    z = df.sort_values(["confidence","external_trial_id"], ascending=[False,True], kind="stable")
    return z.iloc[:max(1, math.ceil(coverage * len(z)))]


def select_pb(df, coverage):
    parts = []
    for _, g in df.groupby("participant_id", observed=True, sort=False):
        z = g.sort_values(["confidence","external_trial_id"], ascending=[False,True], kind="stable")
        parts.append(z.iloc[:max(1, math.ceil(coverage * len(z)))])
    return pd.concat(parts, ignore_index=True)


def select(df, coverage, mode):
    return select_global(df, coverage) if mode == "global" else select_pb(df, coverage)


def counts(full, kept):
    fe = int((full.correct == 0).sum()); fc = int((full.correct == 1).sum())
    re = int((kept.correct == 0).sum()); rc = int((kept.correct == 1).sum())
    we, wc = fe - re, fc - rc
    be = wc / we if we > 0 else (np.nan if wc == 0 else np.inf)
    return dict(n_total=len(full), n_retained=len(kept), n_withheld=len(full)-len(kept),
                full_errors=fe, full_correct=fc, retained_errors=re, retained_correct=rc,
                withheld_errors=we, withheld_correct=wc,
                break_even_lambda_error_over_miss=be)


def observed_tables(long, coverages, lambdas):
    be_rows, curve_rows = [], []
    for (dataset, method), full in long.groupby(["dataset","method"], observed=True, sort=True):
        for mode in ["global", "participant_balanced"]:
            for cov in coverages:
                c = counts(full, select(full, cov, mode))
                be_rows.append(dict(dataset=dataset, method=method, selection_mode=mode,
                                    coverage=cov, analysis_status="posthoc_interaction_cost_sensitivity", **c))
                for lam in lambdas:
                    full_cost = lam * c["full_errors"]
                    sel_cost = lam * c["retained_errors"] + c["withheld_correct"]
                    saving = full_cost - sel_cost
                    curve_rows.append(dict(dataset=dataset, method=method, selection_mode=mode,
                                           coverage=cov, lambda_error_over_miss=lam,
                                           full_cost=full_cost, selective_cost=sel_cost,
                                           absolute_cost_saving=saving,
                                           relative_cost_saving=saving/full_cost if full_cost > 0 else np.nan,
                                           policy_cheaper_than_full=saving > 0))
    return pd.DataFrame(be_rows), pd.DataFrame(curve_rows)


def observed_optimal(curves):
    rows=[]
    for key,g in curves.groupby(["dataset","method","selection_mode","lambda_error_over_miss"], observed=True):
        best=g.sort_values(["selective_cost","coverage"], ascending=[True,False], kind="stable").iloc[0]
        rows.append(dict(dataset=key[0], method=key[1], selection_mode=key[2], lambda_error_over_miss=key[3],
                         optimal_coverage=float(best.coverage), optimal_selective_cost=float(best.selective_cost),
                         full_cost=float(best.full_cost), optimal_relative_cost_saving=float(best.relative_cost_saving)))
    return pd.DataFrame(rows)


class Block:
    def __init__(self, confidence, correct):
        self.conf = np.asarray(confidence, float)
        self.correct = np.asarray(correct, np.int8)
        self.n = len(self.correct)
        o=np.argsort(-self.conf, kind="stable")
        sc=self.correct[o]
        self.cum_e=np.concatenate([[0], np.cumsum(1-sc, dtype=np.int64)])
        self.cum_c=np.concatenate([[0], np.cumsum(sc, dtype=np.int64)])


def make_blocks(df):
    return [Block(g.confidence.to_numpy(float), g.correct.to_numpy(np.int8))
            for _,g in df.groupby("participant_id", observed=True, sort=True)]


def bootstrap_combo(df, coverages, key_lambdas, nboot, rng, dataset, method, progress_every):
    blocks=make_blocks(df); n=len(blocks)
    cost_rows=[]; opt_rows=[]; start=time.perf_counter()
    for b in range(nboot):
        sample=[blocks[int(i)] for i in rng.integers(0,n,size=n)]
        fe=sum(int(np.sum(1-x.correct)) for x in sample)
        fc=sum(int(np.sum(x.correct)) for x in sample)

        conf=np.concatenate([x.conf for x in sample]); cor=np.concatenate([x.correct for x in sample])
        o=np.argsort(-conf, kind="stable"); sc=cor[o]
        ge=np.concatenate([[0],np.cumsum(1-sc,dtype=np.int64)])
        gc=np.concatenate([[0],np.cumsum(sc,dtype=np.int64)])
        nt=len(sc)
        per_mode={"global":{},"participant_balanced":{}}
        for cov in coverages:
            k=max(1,math.ceil(cov*nt)); re=int(ge[k]); rc=int(gc[k])
            per_mode["global"][cov]=(re,rc,fe-re,fc-rc)
            re=rc=0
            for x in sample:
                kp=max(1,math.ceil(cov*x.n)); re+=int(x.cum_e[kp]); rc+=int(x.cum_c[kp])
            per_mode["participant_balanced"][cov]=(re,rc,fe-re,fc-rc)

        for mode in ["global","participant_balanced"]:
            for cov in coverages:
                re,rc,we,wc=per_mode[mode][cov]
                be=wc/we if we>0 else (np.nan if wc==0 else np.inf)
                for lam in key_lambdas:
                    full_cost=lam*fe; sel_cost=lam*re+wc; saving=full_cost-sel_cost
                    cost_rows.append(dict(dataset=dataset,method=method,selection_mode=mode,replicate=b,
                                          coverage=cov,lambda_error_over_miss=lam,
                                          break_even_lambda_error_over_miss=be,
                                          relative_cost_saving=saving/full_cost if full_cost>0 else np.nan,
                                          policy_cheaper_than_full=saving>0))
            for lam in key_lambdas:
                cand=[]
                for cov in coverages:
                    re,rc,we,wc=per_mode[mode][cov]
                    cand.append((lam*re+wc,-cov,cov))
                opt_rows.append(dict(dataset=dataset,method=method,selection_mode=mode,replicate=b,
                                     lambda_error_over_miss=lam,optimal_coverage=min(cand)[2]))
        if progress_every and ((b+1)%progress_every==0 or b+1==nboot):
            elapsed=time.perf_counter()-start; rate=(b+1)/elapsed if elapsed else np.nan
            eta=(nboot-b-1)/rate if np.isfinite(rate) and rate>0 else np.nan
            print(f"[bootstrap] {dataset}/{method}: {b+1}/{nboot} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)
    return pd.DataFrame(cost_rows), pd.DataFrame(opt_rows)


def summarize_boot_cost(reps):
    rows=[]
    keys=["dataset","method","selection_mode","coverage","lambda_error_over_miss"]
    for key,g in reps.groupby(keys, observed=True, sort=True):
        r=g.relative_cost_saving.to_numpy(float)
        be=g.break_even_lambda_error_over_miss.replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
        rows.append(dict(dataset=key[0],method=key[1],selection_mode=key[2],coverage=key[3],lambda_error_over_miss=key[4],
                         relative_cost_saving_mean=float(np.mean(r)),
                         relative_cost_saving_ci_low=float(np.quantile(r,.025)),
                         relative_cost_saving_ci_high=float(np.quantile(r,.975)),
                         bootstrap_probability_policy_cheaper_than_full=float(np.mean(g.policy_cheaper_than_full.to_numpy(bool))),
                         break_even_lambda_median=float(np.median(be)) if len(be) else np.nan,
                         break_even_lambda_ci_low=float(np.quantile(be,.025)) if len(be) else np.nan,
                         break_even_lambda_ci_high=float(np.quantile(be,.975)) if len(be) else np.nan,
                         n_bootstrap=len(g)))
    return pd.DataFrame(rows)


def summarize_boot_opt(reps):
    rows=[]
    keys=["dataset","method","selection_mode","lambda_error_over_miss"]
    for key,g in reps.groupby(keys, observed=True, sort=True):
        vc=g.optimal_coverage.value_counts(normalize=True)
        mp=vc.max(); modal=max(float(c) for c,p in vc.items() if p==mp)
        rows.append(dict(dataset=key[0],method=key[1],selection_mode=key[2],lambda_error_over_miss=key[3],
                         modal_optimal_coverage=modal,modal_probability=float(vc.loc[modal]),
                         prob_full_coverage_optimal=float(vc.get(1.0,0)),prob_50pct_optimal=float(vc.get(.5,0)),
                         prob_20pct_optimal=float(vc.get(.2,0)),prob_10pct_optimal=float(vc.get(.1,0)),n_bootstrap=len(g)))
    return pd.DataFrame(rows)


def plots(curves, be, outdir):
    import matplotlib.pyplot as plt
    for dataset in ["English","Turkish"]:
        g=curves[(curves.dataset==dataset)&(curves.method==PRIMARY)&(curves.selection_mode=="participant_balanced")&curves.coverage.isin([.5,.2,.1])]
        if len(g):
            fig,ax=plt.subplots(figsize=(8,5))
            for cov,h in g.groupby("coverage",observed=True):
                h=h.sort_values("lambda_error_over_miss")
                ax.plot(h.lambda_error_over_miss,h.relative_cost_saving,label=f"{int(cov*100)}% coverage")
            ax.axhline(0,linewidth=1); ax.set_xscale("log")
            ax.set_xlabel("Cost ratio λ = incorrect-intervention cost / missed-opportunity cost")
            ax.set_ylabel("Relative cost saving vs full coverage")
            ax.set_title(f"{dataset}: participant-balanced QCN-PC cost sensitivity")
            ax.legend(); fig.tight_layout(); fig.savefig(outdir/f"{dataset.lower()}_participant_balanced_cost_sensitivity.png",dpi=200,bbox_inches="tight"); plt.close(fig)
    p=be[(be.selection_mode=="participant_balanced")&(be.coverage<1)&np.isfinite(be.break_even_lambda_error_over_miss)]
    fig,ax=plt.subplots(figsize=(9,5.5))
    for (dataset,method),g in p.groupby(["dataset","method"],observed=True):
        g=g.sort_values("coverage"); ax.plot(g.coverage,g.break_even_lambda_error_over_miss,marker="o",label=f"{dataset}/{method}")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Break-even λ")
    ax.set_title("When does abstention become cheaper than full coverage?")
    ax.legend(); fig.tight_layout(); fig.savefig(outdir/"participant_balanced_break_even_lambda.png",dpi=200,bbox_inches="tight"); plt.close(fig)


def main():
    a=parse_args(); coverages=sorted(set(flist(a.coverages)),reverse=True)
    if 1.0 not in coverages: coverages.insert(0,1.0)
    key_lambdas=sorted(set(flist(a.key_lambdas)))
    dense=np.unique(np.concatenate([np.logspace(np.log10(a.lambda_min),np.log10(a.lambda_max),a.lambda_grid_points),np.asarray(key_lambdas)]))

    print("[1/7] Loading frozen English/Turkish scores...", flush=True)
    long,threshold=load_data(a.english_predictions,a.turkish_predictions)
    print("[2/7] Computing observed break-even ratios...", flush=True)
    be,curves=observed_tables(long,coverages,dense)
    print("[3/7] Computing optimal coverage across λ...", flush=True)
    opt=observed_optimal(curves)
    print(f"[4/7] Participant bootstrap: {a.bootstrap} replicates per dataset/method...", flush=True)
    cparts=[]; oparts=[]; idx=0
    for dataset in ["English","Turkish"]:
        for method in [REFERENCE,PRIMARY]:
            df=long[(long.dataset==dataset)&(long.method==method)].copy()
            c,o=bootstrap_combo(df,coverages,key_lambdas,a.bootstrap,np.random.default_rng(a.seed+idx*100003),dataset,method,a.progress_every)
            idx+=1; cparts.append(c); oparts.append(o)
    breps=pd.concat(cparts,ignore_index=True); oreps=pd.concat(oparts,ignore_index=True)
    print("[5/7] Summarizing bootstrap uncertainty...", flush=True)
    bsum=summarize_boot_cost(breps); osum=summarize_boot_opt(oreps)
    print("[6/7] Writing CSV outputs...", flush=True)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    be.to_csv(a.output_dir/"interaction_cost_break_even.csv",index=False,encoding="utf-8-sig")
    curves.to_csv(a.output_dir/"interaction_cost_lambda_curves.csv",index=False,encoding="utf-8-sig")
    curves[curves.lambda_error_over_miss.isin(key_lambdas)].to_csv(a.output_dir/"interaction_cost_key_ratios.csv",index=False,encoding="utf-8-sig")
    opt.to_csv(a.output_dir/"optimal_coverage_by_lambda.csv",index=False,encoding="utf-8-sig")
    bsum.to_csv(a.output_dir/"interaction_cost_bootstrap_summary.csv",index=False,encoding="utf-8-sig")
    osum.to_csv(a.output_dir/"optimal_coverage_bootstrap_summary.csv",index=False,encoding="utf-8-sig")
    print("[7/7] Creating figures and manifest...", flush=True)
    plots(curves,be,a.output_dir)
    manifest={
        "analysis":"POST-HOC interaction-cost ratio sensitivity",
        "primary_policy_score":PRIMARY,
        "reference_policy_score":REFERENCE,
        "frozen_threshold":threshold,
        "cost_model":"L = C_error * retained_errors + C_miss * withheld_correct_predictions",
        "lambda":"C_error / C_miss; normalize C_miss=1",
        "break_even":"lambda* = withheld_correct / withheld_errors; selection cheaper iff lambda > lambda*",
        "coverages":coverages,"key_lambdas":key_lambdas,"bootstrap":a.bootstrap,"seed":a.seed,
        "status":"Post-hoc. Turkish 100/50/20 coverage points were previously locked, but this cost function was not pre-specified.",
        "boundary":"Offline policy-cost proxy only; no claim about actual comprehension, frustration, trust, delay cost, or intervention utility."
    }
    (a.output_dir/"interaction_cost_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

    print("\n=== OBSERVED BREAK-EVEN COST RATIOS ===")
    k=be[be.coverage.isin([.5,.2,.1])]
    print(k[["dataset","method","selection_mode","coverage","withheld_errors","withheld_correct","break_even_lambda_error_over_miss"]].to_string(index=False))
    print("\n=== BOOTSTRAP: PARTICIPANT-BALANCED QCN-PC ===")
    k=bsum[(bsum.method==PRIMARY)&(bsum.selection_mode=="participant_balanced")&bsum.coverage.isin([.5,.2,.1])]
    print(k[["dataset","coverage","lambda_error_over_miss","relative_cost_saving_mean","relative_cost_saving_ci_low","relative_cost_saving_ci_high","bootstrap_probability_policy_cheaper_than_full","break_even_lambda_median","break_even_lambda_ci_low","break_even_lambda_ci_high"]].to_string(index=False))
    print("\n=== BOOTSTRAP MODAL OPTIMAL COVERAGE: PARTICIPANT-BALANCED QCN-PC ===")
    k=osum[(osum.method==PRIMARY)&(osum.selection_mode=="participant_balanced")]
    print(k[["dataset","lambda_error_over_miss","modal_optimal_coverage","modal_probability","prob_full_coverage_optimal","prob_50pct_optimal","prob_20pct_optimal","prob_10pct_optimal"]].to_string(index=False))
    print(f"\nSaved: {a.output_dir.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
