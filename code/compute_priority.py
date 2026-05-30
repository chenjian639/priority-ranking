"""
Compute observation priority for PC/APC candidates.
Reads processed/train_modeling_table_cp_kp_vs_fp_fa.csv to train a classifier,
then scores processed/candidate_pc_apc_for_risk_prediction.csv and
writes ranking outputs to processed/.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

WORKDIR = Path(__file__).resolve().parents[1]
PROCESSED = WORKDIR / 'processed'
PRIORITY_DIR = PROCESSED / 'priority_outputs'
PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = PROCESSED / 'train_modeling_table_cp_kp_vs_fp_fa.csv'
CAND_PATH = PROCESSED / 'candidate_pc_apc_for_risk_prediction.csv'

OUT_ALL = PRIORITY_DIR / 'candidates_with_scores.csv'
OUT_TOP = PRIORITY_DIR / 'top20_priority.csv'
OUT_HIGH_RISK = PRIORITY_DIR / 'top20_high_risk.csv'
MODEL_PATH = PRIORITY_DIR / 'planet_rf_model.joblib'

FEATURES = ['pl_orbper', 'pl_rade', 'st_tmag', 'st_teff', 'st_rad', 'missing_feature_count']
RANDOM_STATE = 42


def safe_minmax(series):
    s = series.fillna(series.median())
    mn = s.min()
    mx = s.max()
    if mx - mn == 0:
        return s, mn, mx
    return (s - mn) / (mx - mn), mn, mx


def compute_scores(df, planet_prob, feature_ranges):
    # brightness: lower st_tmag is better
    st_tmag = df['st_tmag'].fillna(feature_ranges['st_tmag_med'])
    pl_orbper = df['pl_orbper'].fillna(feature_ranges['pl_orbper_med'])
    pl_rade = df['pl_rade'].fillna(feature_ranges['pl_rade_med'])
    missing = df['missing_feature_count'].fillna(0)

    # normalized scores (0..1)
    st_tmag_norm = (feature_ranges['st_tmag_max'] - st_tmag) / (feature_ranges['st_tmag_max'] - feature_ranges['st_tmag_min']) if feature_ranges['st_tmag_max']!=feature_ranges['st_tmag_min'] else 0.5
    short_period_norm = (feature_ranges['pl_orbper_max'] - pl_orbper) / (feature_ranges['pl_orbper_max'] - feature_ranges['pl_orbper_min']) if feature_ranges['pl_orbper_max']!=feature_ranges['pl_orbper_min'] else 0.5

    # science interest: similarity to Earth in radius and incident flux
    # approximate stellar luminosity (L/L_sun) and insolation (S in Earth flux)
    # use simple relations; fall back to medians when values missing
    st_rad = df['st_rad'].fillna(feature_ranges.get('st_rad_med', feature_ranges['st_rad_min']))
    st_teff = df['st_teff'].fillna(feature_ranges.get('st_teff_med', 5772.0))
    P_days = df['pl_orbper'].fillna(feature_ranges.get('pl_orbper_med', 365.0))
    pl_r = df['pl_rade'].fillna(feature_ranges.get('pl_rade_med', 1.0))

    # compute approximate luminosity in L_sun
    T_sun = 5772.0
    L = (st_rad ** 2) * (st_teff / T_sun) ** 4

    # approximate stellar mass from radius (main-sequence rough): M ~ R^0.8 (in solar units)
    M = st_rad ** 0.8

    # Kepler to get semi-major axis a (AU): a^3 = G*M*(P_seconds^2)/(4*pi^2)
    # constants
    G = 6.67430e-11
    M_sun = 1.98847e30
    AU = 1.495978707e11

    P_sec = P_days * 86400.0
    # avoid invalid values
    with np.errstate(invalid='ignore'):
        a_m = ((G * (M * M_sun) * (P_sec ** 2)) / (4.0 * (np.pi ** 2))) ** (1.0/3.0)
    a_AU = a_m / AU
    # when computation fails or gives nan/inf, fallback to large value
    a_AU = a_AU.fillna(feature_ranges.get('a_med', 1.0)).replace([np.inf, -np.inf], feature_ranges.get('a_med', 1.0))

    S = L / (a_AU ** 2)

    # similarity kernels (log-normal-like) centered at 1 (Earth)
    # use reasonable bandwidths
    def similarity_to_one(x, sigma=0.5):
        # x may be array-like; use log-space Gaussian
        x_safe = np.clip(x, 1e-6, None)
        return np.exp(-0.5 * (np.log(x_safe) / sigma) ** 2)

    radius_similarity = similarity_to_one(pl_r, sigma=0.25)
    insolation_similarity = similarity_to_one(S, sigma=0.5)

    science_interest = 0.6 * radius_similarity + 0.4 * insolation_similarity

    # combine into priority: place more weight on model probability and science interest,
    # reduce raw brightness/short-period bias
    priority_score = (0.5 * planet_prob + 0.2 * science_interest + 0.15 * st_tmag_norm + 0.1 * short_period_norm)

    # penalty for missing features (normalized)
    max_missing = feature_ranges.get('max_missing', 5)
    missing_penalty = 0.05 * (missing / max_missing)
    priority_score = priority_score - missing_penalty
    # clip
    priority_score = np.clip(priority_score, 0.0, 1.0)
    return priority_score


def main():
    print('Loading training data...')
    train = pd.read_csv(TRAIN_PATH)
    print('Loading candidate data...')
    cand = pd.read_csv(CAND_PATH)

    # prepare features
    X_train = train[FEATURES].copy()
    y_train = train['target_confirmed_planet']

    # pipeline: impute + scaler
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)

    # train model
    print('Training RandomForest model...')
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    # save pipeline and model
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'model': rf}, MODEL_PATH)
    print('Saved model to', MODEL_PATH)

    # transform candidates
    X_cand = cand[FEATURES].copy()
    X_cand_imp = imputer.transform(X_cand)
    X_cand_scaled = scaler.transform(X_cand_imp)

    planet_prob = rf.predict_proba(X_cand_scaled)[:, 1]

    # feature ranges for normalization
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
        'max_missing': int(max(1, train['missing_feature_count'].max()))
    }

    # add stellar and temperature medians/mins for science score fallbacks
    feature_ranges['st_rad_min'] = float(train['st_rad'].min(skipna=True)) if 'st_rad' in train.columns else 1.0
    feature_ranges['st_rad_med'] = float(train['st_rad'].median(skipna=True)) if 'st_rad' in train.columns else 1.0
    feature_ranges['st_teff_med'] = float(train['st_teff'].median(skipna=True)) if 'st_teff' in train.columns else 5772.0

    # approximate median semi-major axis (AU) from train using same approximation used in compute_scores
    try:
        T_sun = 5772.0
        G = 6.67430e-11
        M_sun = 1.98847e30
        AU = 1.495978707e11
        st_rad_train = train['st_rad'].fillna(feature_ranges['st_rad_med'])
        st_teff_train = train['st_teff'].fillna(feature_ranges['st_teff_med'])
        P_days_train = train['pl_orbper'].fillna(feature_ranges['pl_orbper_med'])
        M_train = st_rad_train ** 0.8
        P_sec_train = P_days_train * 86400.0
        with np.errstate(invalid='ignore'):
            a_m_train = ((G * (M_train * M_sun) * (P_sec_train ** 2)) / (4.0 * (np.pi ** 2))) ** (1.0/3.0)
        a_AU_train = (a_m_train / AU).replace([np.inf, -np.inf], np.nan)
        feature_ranges['a_med'] = float(np.nanmedian(a_AU_train)) if not np.all(np.isnan(a_AU_train)) else 1.0
    except Exception:
        feature_ranges['a_med'] = 1.0

    cand = cand.reset_index(drop=True)
    cand['planet_probability'] = planet_prob
    cand['priority_score'] = compute_scores(cand, cand['planet_probability'], feature_ranges)

    # save full
    cand.to_csv(OUT_ALL, index=False)
    print('Wrote candidates with scores to', OUT_ALL)

    top = cand.sort_values('priority_score', ascending=False).head(20)
    top.to_csv(OUT_TOP, index=False)
    print('Wrote top 20 priority to', OUT_TOP)

    high_risk = cand.sort_values('planet_probability', ascending=True).head(20)
    high_risk.to_csv(OUT_HIGH_RISK, index=False)
    print('Wrote top 20 high-risk (low prob) to', OUT_HIGH_RISK)


if __name__ == '__main__':
    main()
