"""
Cross-validation evaluation of the planet classifier and
breakdown + visualization for Top20 priority candidates.

Outputs (to processed/):
- cv_metrics.txt
- calibration_curve.png
- top20_breakdown.csv
- top20_contributions.png
- radius_insolation_scatter.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
import matplotlib.pyplot as plt
import joblib

WORKDIR = Path(__file__).resolve().parents[1]
PROCESSED = WORKDIR / 'processed'
PRIORITY_DIR = PROCESSED / 'priority_outputs'
PRIORITY_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED / 'train_modeling_table_cp_kp_vs_fp_fa.csv'
CAND_PATH = PROCESSED / 'candidate_pc_apc_for_risk_prediction.csv'
MODEL_PATH = PRIORITY_DIR / 'planet_rf_model.joblib'

FEATURES = ['pl_orbper', 'pl_rade', 'st_tmag', 'st_teff', 'st_rad', 'missing_feature_count']

def compute_science_interest(df, feature_ranges):
    st_rad = df['st_rad'].fillna(feature_ranges.get('st_rad_med', 1.0))
    st_teff = df['st_teff'].fillna(feature_ranges.get('st_teff_med', 5772.0))
    P_days = df['pl_orbper'].fillna(feature_ranges.get('pl_orbper_med', 365.0))
    pl_r = df['pl_rade'].fillna(feature_ranges.get('pl_rade_med', 1.0))
    T_sun = 5772.0
    L = (st_rad ** 2) * (st_teff / T_sun) ** 4
    M = st_rad ** 0.8
    # constants
    G = 6.67430e-11
    M_sun = 1.98847e30
    AU = 1.495978707e11
    P_sec = P_days * 86400.0
    with np.errstate(invalid='ignore'):
        a_m = ((G * (M * M_sun) * (P_sec ** 2)) / (4.0 * (np.pi ** 2))) ** (1.0/3.0)
    a_AU = a_m / AU
    a_AU = a_AU.fillna(feature_ranges.get('a_med', 1.0)).replace([np.inf, -np.inf], feature_ranges.get('a_med', 1.0))
    S = L / (a_AU ** 2)
    def sim(x, sigma=0.5):
        x_safe = np.clip(x, 1e-6, None)
        return np.exp(-0.5 * (np.log(x_safe) / sigma) ** 2)
    radius_sim = sim(pl_r, sigma=0.25)
    insol_sim = sim(S, sigma=0.5)
    return 0.6 * radius_sim + 0.4 * insol_sim, S

def main():
    train = pd.read_csv(TRAIN_PATH)
    X = train[FEATURES].copy()
    y = train['target_confirmed_planet']

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.zeros(len(train))
    preds = np.zeros(len(train), dtype=int)

    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        Xtr = X.iloc[tr_idx]
        Xte = X.iloc[te_idx]
        ytr = y.iloc[tr_idx]
        yte = y.iloc[te_idx]

        imp = SimpleImputer(strategy='median')
        sc = StandardScaler()
        Xtr_imp = imp.fit_transform(Xtr)
        Xtr_s = sc.fit_transform(Xtr_imp)
        Xte_imp = imp.transform(Xte)
        Xte_s = sc.transform(Xte_imp)

        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(Xtr_s, ytr)
        p = rf.predict_proba(Xte_s)[:,1]
        pr = (p >= 0.5).astype(int)

        probs[te_idx] = p
        preds[te_idx] = pr

        auc = roc_auc_score(yte, p)
        brier = brier_score_loss(yte, p)
        acc = accuracy_score(yte, pr)
        fold_metrics.append((fold, auc, brier, acc))

    # aggregate
    overall_auc = roc_auc_score(y, probs)
    overall_brier = brier_score_loss(y, probs)
    overall_acc = accuracy_score(y, (probs>=0.5).astype(int))

    # save metrics
    with open(PRIORITY_DIR / 'cv_metrics.txt', 'w') as f:
        f.write('Fold,AUC,Brier,Accuracy\n')
        for fm in fold_metrics:
            f.write(f'{fm[0]},{fm[1]:.4f},{fm[2]:.4f},{fm[3]:.4f}\n')
        f.write(f'Overall,{overall_auc:.4f},{overall_brier:.4f},{overall_acc:.4f}\n')

    # calibration curve (simple reliability diagram)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Reliability')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.title('Calibration curve')
    plt.savefig(PRIORITY_DIR / 'calibration_curve.png', dpi=150)

    # retrain on full training set and save model pipeline
    X_imp_full = imputer.fit_transform(X)
    X_s_full = scaler.fit_transform(X_imp_full)
    rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_full.fit(X_s_full, y)
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'model': rf_full}, MODEL_PATH)

    # load candidates and compute breakdown for Top20
    cand = pd.read_csv(CAND_PATH).reset_index(drop=True)
    # apply imputer/scaler from full training
    X_cand = cand[FEATURES].copy()
    X_cand_imp = imputer.transform(X_cand)
    X_cand_s = scaler.transform(X_cand_imp)
    cand['planet_probability'] = rf_full.predict_proba(X_cand_s)[:,1]

    # prepare feature ranges for normalization
    feature_ranges = {
        'st_tmag_min': float(train['st_tmag'].min()),
        'st_tmag_max': float(train['st_tmag'].max()),
        'st_tmag_med': float(train['st_tmag'].median()),
        'pl_orbper_min': float(train['pl_orbper'].min(skipna=True)),
        'pl_orbper_max': float(train['pl_orbper'].max(skipna=True)),
        'pl_orbper_med': float(train['pl_orbper'].median(skipna=True)),
        'pl_rade_min': float(train['pl_rade'].min(skipna=True)),
        'pl_rade_max': float(train['pl_rade'].max(skipna=True)),
        'pl_rade_med': float(train['pl_rade'].median(skipna=True)),
        'max_missing': int(max(1, train['missing_feature_count'].max())),
        'st_rad_min': float(train['st_rad'].min(skipna=True)),
        'st_rad_med': float(train['st_rad'].median(skipna=True)),
        'st_teff_med': float(train['st_teff'].median(skipna=True))
    }
    # compute science interest and insolation
    sci, S = compute_science_interest(cand, feature_ranges)
    cand['science_interest'] = sci
    cand['insolation'] = S

    # normalized components
    st_tmag = cand['st_tmag'].fillna(feature_ranges['st_tmag_med'])
    st_tmag_norm = (feature_ranges['st_tmag_max'] - st_tmag) / (feature_ranges['st_tmag_max'] - feature_ranges['st_tmag_min'])
    pl_orbper = cand['pl_orbper'].fillna(feature_ranges['pl_orbper_med'])
    short_period_norm = (feature_ranges['pl_orbper_max'] - pl_orbper) / (feature_ranges['pl_orbper_max'] - feature_ranges['pl_orbper_min'])
    missing = cand['missing_feature_count'].fillna(0)

    # contributions
    w = {'planet':0.5, 'science':0.2, 'brightness':0.15, 'period':0.1}
    cand['contrib_planet'] = cand['planet_probability'] * w['planet']
    cand['contrib_science'] = cand['science_interest'] * w['science']
    cand['contrib_brightness'] = st_tmag_norm * w['brightness']
    cand['contrib_period'] = short_period_norm * w['period']
    max_missing = feature_ranges.get('max_missing',5)
    cand['missing_penalty'] = 0.05 * (missing / max_missing)
    cand['priority_score_calc'] = cand['contrib_planet'] + cand['contrib_science'] + cand['contrib_brightness'] + cand['contrib_period'] - cand['missing_penalty']
    cand['priority_score_calc'] = np.clip(cand['priority_score_calc'], 0.0, 1.0)

    top20 = cand.sort_values('priority_score_calc', ascending=False).head(20).copy()
    top20.to_csv(PRIORITY_DIR / 'top20_breakdown.csv', index=False)

    # stacked bar plot of contributions
    plt.figure(figsize=(10,6))
    inds = np.arange(len(top20))
    labels = top20['toidisplay'].astype(str)
    p1 = plt.bar(inds, top20['contrib_planet'], label='planet_prob')
    p2 = plt.bar(inds, top20['contrib_science'], bottom=top20['contrib_planet'], label='science_interest')
    p3 = plt.bar(inds, top20['contrib_brightness'], bottom=top20['contrib_planet']+top20['contrib_science'], label='brightness')
    p4 = plt.bar(inds, top20['contrib_period'], bottom=top20['contrib_planet']+top20['contrib_science']+top20['contrib_brightness'], label='period')
    # subtract penalty as red marker
    plt.scatter(inds, top20['priority_score_calc'] - top20['missing_penalty'], color='red', zorder=5, label='score (before penalty)')
    plt.xticks(inds, labels, rotation=90)
    plt.ylabel('Contribution to priority score')
    plt.title('Top20 priority contributions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PRIORITY_DIR / 'top20_contributions.png', dpi=150)

    # scatter radius vs insolation colored by planet_probability
    plt.figure(figsize=(6,5))
    sc = plt.scatter(cand['pl_rade'], cand['insolation'], c=cand['planet_probability'], cmap='viridis', s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar(sc, label='planet_probability')
    plt.xlabel('pl_rade (R_earth)')
    plt.ylabel('insolation (Earth flux)')
    plt.title('Radius vs Insolation colored by planet_probability')
    plt.tight_layout()
    plt.savefig(PRIORITY_DIR / 'radius_insolation_scatter.png', dpi=150)

    print('Analysis complete. Outputs saved to processed/')

if __name__ == '__main__':
    main()
