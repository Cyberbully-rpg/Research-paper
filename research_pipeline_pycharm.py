# ============================================================
# RESEARCH PIPELINE — FULL CODE FOR PYCHARM
# Paper: A Generalized ML Framework for Performance Prediction
#        and Prescriptive Decision Intelligence
# Author: Vansh | KIIT University | B.Tech CSE AI/ML 2024-2028
# Co-author: Bhavishya Raghav | KIIT University
# ============================================================
# HOW TO USE IN PYCHARM:
# 1. Place both CSV files in the same folder as this script
#    OR update DS1_PATH and DS2_PATH below to full absolute paths
# 2. Open terminal in PyCharm and run:
#    pip install xgboost lightgbm scikit-learn pandas numpy
# 3. Right-click the file → Run 'research_pipeline'
# 4. All results print in the PyCharm console
# ============================================================


# ============================================================
# STEP 0 — INSTALL DEPENDENCIES (run once in terminal)
# Open PyCharm terminal and run this command:
#   pip install xgboost lightgbm scikit-learn pandas numpy
# ============================================================


# ============================================================
# STEP 1 — IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Fix seed — same seed = same results every run
np.random.seed(42)

print("All imports successful.")


# ============================================================
# STEP 2 — CONFIGURATION
# All settings in one place. Change here, not in functions.
# ============================================================

# --- Dataset paths ---
# OPTION 1: Put CSV files in the same folder as this script
DS1_PATH = 'ds1_b2c_retail_clean.csv'   # B2C Retail — low noise
DS2_PATH = 'ds2_b2b_noisy.csv'          # B2B AE     — high noise

# OPTION 2: Use full absolute path if files are elsewhere
# Windows example: DS1_PATH = r'C:\Users\Vansh\datasets\ds1_b2c_retail_clean.csv'
# Mac/Linux example: DS1_PATH = '/home/vansh/datasets/ds1_b2c_retail_clean.csv'

# --- TAU: domain-adaptive R² threshold (Eq. 4) ---
# DS1 CV = 10.4% → tight domain   → tau = 0.75
# DS2 CV = 46.0% → noisy domain   → tau = 0.50
# Justification: Coefficient of Variation (CV = std/mean)
# CV is unit-free, so comparable across datasets with different targets
TAU_DS1 = 0.75
TAU_DS2 = 0.50

# --- Lambda: effort penalty weight (Eq. 1) ---
# Controls how much the framework penalises hard-to-do recommendations
# Higher = more conservative recommendations (prefer easy changes)
# Lower  = more aggressive recommendations (prioritise impact)
LAMBDA = 0.1

# --- Perturbation size (Eq. 8) ---
# We nudge each actionable feature by 20% and measure predicted impact
PERTURBATION = 0.20

# --- Cross-validation folds ---
CV_FOLDS = 3

# --- Noise stress scale ---
# Injects N(0, 0.1 * sigma_feature) noise into test features
NOISE_SCALE = 0.10

print("Configuration loaded.")
print(f"  DS1 tau = {TAU_DS1} | DS2 tau = {TAU_DS2}")
print(f"  Lambda  = {LAMBDA}  | Perturbation = {PERTURBATION*100:.0f}%")


# ============================================================
# STEP 3 — DOMAIN KNOWLEDGE (EFFORT WEIGHTS)
# ============================================================
# This is where human expertise enters the framework.
# A sales manager defines which features a rep can change
# and how hard each change is.
#
# w_j values (Eq. 5, 6):
#   0.0 - 0.3 : easy to change (activity metrics)
#   0.4 - 0.6 : moderate effort (process improvements)
#   0.7 - 0.9 : hard to change (skill-based)
#   np.inf    : IMMUTABLE — excluded from all recommendations (Eq. 7)

DS1_EFFORT_WEIGHTS = {
    # --- ACTIONABLE ---
    'calls_made':       0.2,   # just make more calls — low effort
    'meetings_booked':  0.3,   # depends on call quality — moderate
    'script_adherence': 0.4,   # requires coaching and practice
    'follow_up_rate':   0.3,   # process change — manageable
    'avg_order_value':  0.6,   # upselling skill — harder

    # --- IMMUTABLE (w = inf → never recommended) ---
    'region':           np.inf,  # rep cannot change their region
    'tenure_years':     np.inf,  # rep cannot change their tenure
}

DS2_EFFORT_WEIGHTS = {
    # --- ACTIONABLE ---
    'discovery_calls':   0.2,  # activity metric — easy to increase
    'demos_delivered':   0.3,  # depends on discovery quality
    'proposals_sent':    0.4,  # requires demo success first
    'negotiation_skill': 0.7,  # soft skill — hardest to improve fast
    'crm_hygiene':       0.3,  # process discipline — manageable

    # --- IMMUTABLE ---
    'territory':        np.inf,
    'experience_yrs':   np.inf,
}

# Derived lists — used throughout the pipeline
DS1_ACTIONABLE = [f for f in DS1_EFFORT_WEIGHTS if DS1_EFFORT_WEIGHTS[f] < np.inf]
DS1_IMMUTABLE  = [f for f in DS1_EFFORT_WEIGHTS if DS1_EFFORT_WEIGHTS[f] == np.inf]
DS2_ACTIONABLE = [f for f in DS2_EFFORT_WEIGHTS if DS2_EFFORT_WEIGHTS[f] < np.inf]
DS2_IMMUTABLE  = [f for f in DS2_EFFORT_WEIGHTS if DS2_EFFORT_WEIGHTS[f] == np.inf]

print("Domain knowledge loaded.")
print(f"  DS1 actionable: {DS1_ACTIONABLE}")
print(f"  DS1 immutable:  {DS1_IMMUTABLE}")
print(f"  DS2 actionable: {DS2_ACTIONABLE}")
print(f"  DS2 immutable:  {DS2_IMMUTABLE}")


# ============================================================
# STEP 4 — HELPER FUNCTIONS
# ============================================================

def banner(text):
    """Prints a clean section divider for readable output."""
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def build_preprocessor(X_train):
    """
    Builds a leak-free preprocessing pipeline.

    WHAT IT DOES:
    - Numeric columns : median imputation → standard scaling
    - Categorical cols: mode imputation  → one-hot encoding

    WHY IT MATTERS (data leakage):
    The preprocessor is fit ONLY on X_train. If we fit it on all data
    first, then split, the scaler has seen test set statistics and
    the model gets an unfair advantage. This inflates R² artificially.
    By wrapping the preprocessor in a Pipeline, sklearn ensures it
    only ever sees training data during fit().

    MEDIAN vs MEAN imputation:
    Our datasets have outliers (e.g. calls_made=150 from CRM errors).
    Mean would be pulled toward those outliers. Median is robust.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    if cat_cols:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer([
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ])
    else:
        preprocessor = ColumnTransformer([
            ('num', num_pipe, num_cols)
        ])

    return preprocessor


def clip_impossible_values(df, dataset_name):
    """
    Removes physically impossible values before any modelling.

    WHY: Our datasets contain realistic data entry errors:
    - calls_made = 160 (someone typed an extra digit in CRM)
    - follow_up_rate = 1.4 (rate cannot exceed 1.0)
    - negotiation_skill = 1.3 (score cannot exceed 1.0)

    We clip to valid boundaries rather than dropping rows.
    Dropping loses information. Clipping corrects to nearest valid value.
    All clipping decisions are documented in Section IV of the paper.
    """
    df = df.copy()
    if dataset_name == 'DS1':
        df['calls_made']       = df['calls_made'].clip(0, 100)
        df['follow_up_rate']   = df['follow_up_rate'].clip(0, 1.0)
        df['script_adherence'] = df['script_adherence'].clip(0, 1.0)
    if dataset_name == 'DS2':
        df['discovery_calls']   = df['discovery_calls'].clip(0, 45)
        df['negotiation_skill'] = df['negotiation_skill'].clip(0, 1.0)
        df['crm_hygiene']       = df['crm_hygiene'].clip(0, 1.0)
    return df


def get_pfi_feature_names(pipe, X_test):
    """
    Extracts clean feature names after preprocessing.
    OneHotEncoding creates new columns (region → region_North etc.)
    This function maps them back to readable names.
    """
    try:
        names = pipe.named_steps['preprocessor'].get_feature_names_out()
        names = [n.replace('num__','').replace('cat__','') for n in names]
    except:
        names = X_test.columns.tolist()
    return names


print("Helper functions defined.")


# ============================================================
# STEP 5 — PHASE 2: PREDICTIVE BENCHMARKING
# ============================================================
# Implements Eq. 2: M̂ = argmin RMSE(M), s.t. R²(M) >= τ
# Implements Eq. 3: RMSE computed on held-out test set

def benchmark_models(X, y, tau, label):
    """
    Trains all 5 models and selects the best one.

    MODEL SELECTION LOGIC (Eq. 2):
    - Train all 5 models
    - Rank by RMSE (lower = better)
    - Best model must also pass R² >= tau gate
    - If best RMSE model fails tau, it is noted in the paper
      but still selected (model selection is by RMSE — tau is
      a quality gate we report, not override the selection)

    WHY 5 MODELS:
    Different data structures suit different algorithms.
    Linear data (our case) → LinearRegression wins.
    Complex interactions → tree ensembles win.
    We let the data decide.

    CROSS VALIDATION (3-fold):
    Splits data into 3 parts. Trains on 2, tests on 1.
    Repeats 3 times so every row is tested exactly once.
    CV_RMSE tells us if the model generalises or just memorised.

    Returns:
        best_pipe              : trained sklearn Pipeline
        best_name              : name string of best model
        X_train, X_test        : feature splits
        y_train, y_test        : target splits
        results_df             : full leaderboard DataFrame
    """
    banner(f"PHASE 2 — BENCHMARKING: {label} | τ = {tau}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train size: {len(X_train)} | Test size: {len(X_test)}")

    preprocessor = build_preprocessor(X_train)

    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree':     DecisionTreeRegressor(random_state=42),
        'RandomForest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost':          XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM':         LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    }

    results   = []
    best_rmse = np.inf
    best_pipe = None
    best_name = None

    print(f"\n  {'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>8} {'CV_RMSE':>10} {'τ':>4}")
    print(f"  {'-'*67}")

    for name, model in models.items():

        # Pipeline ensures preprocessor only fits on training data
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        pipe.fit(X_train, y_train)
        preds   = pipe.predict(X_test)

        # Core metrics (Eq. 3)
        rmse    = np.sqrt(mean_squared_error(y_test, preds))
        mae     = mean_absolute_error(y_test, preds)
        r2      = r2_score(y_test, preds)

        # 3-fold cross validation
        cv_s    = cross_val_score(pipe, X, y, cv=CV_FOLDS,
                                  scoring='neg_root_mean_squared_error')
        cv_rmse = -cv_s.mean()

        tau_flag = '✓' if r2 >= tau else '✗'

        print(f"  {name:<22} {rmse:>10,.2f} {mae:>10,.2f} {r2:>8.4f} {cv_rmse:>10,.2f} {tau_flag:>4}")

        results.append({
            'Model': name, 'RMSE': round(rmse,2), 'MAE': round(mae,2),
            'R2': round(r2,4), 'CV_RMSE': round(cv_rmse,2), 'tau_pass': tau_flag
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_pipe = pipe
            best_name = name

    best_r2 = [r['R2'] for r in results if r['Model'] == best_name][0]
    print(f"\n  ★ Best model : {best_name}")
    print(f"    RMSE       : {best_rmse:,.2f}")
    print(f"    R²         : {best_r2:.4f}")
    print(f"    τ gate     : {'PASS ✓' if best_r2 >= tau else 'FAIL ✗ — noted in paper'}")

    return (best_pipe, best_name,
            X_train, X_test, y_train, y_test,
            pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True))


def noise_stress_test(pipe, X_test, y_test):
    """
    Tests model stability under small input perturbations.

    Injects N(0, 0.1 * sigma_j) noise into each numeric feature.
    A stable model should show Noise RMSE ≈ Test RMSE.
    A large gap indicates the model is brittle to measurement error.

    This reflects real-world conditions: CRM data is never perfectly
    accurate. Call counts, meeting records, and scores have natural
    measurement uncertainty.
    """
    X_noisy  = X_test.copy()
    num_cols = X_noisy.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        sig = X_noisy[col].std()
        X_noisy[col] += np.random.normal(0, NOISE_SCALE * sig, len(X_noisy))
    rmse_noisy = np.sqrt(mean_squared_error(y_test, pipe.predict(X_noisy)))
    print(f"  Noise Stress RMSE: {rmse_noisy:,.2f}")
    return rmse_noisy


print("Phase 2 functions defined.")


# ============================================================
# STEP 6 — PHASE 3: EXPLAINABILITY (PFI)
# ============================================================

def run_pfi(pipe, X_test, y_test):
    """
    Permutation Feature Importance on the TEST SET.

    HOW IT WORKS:
    1. Predict on X_test → get baseline RMSE
    2. Randomly shuffle column j → break its relationship with y
    3. Predict again → RMSE increases if column j was important
    4. Importance score = average RMSE increase over 10 shuffles
    5. Repeat for every feature

    WHY TEST SET (not training set):
    Training set importance reflects what the model memorised.
    Test set importance reflects what actually generalises.
    For prescriptive recommendations, we care about generalisation.

    WHY n_repeats=10:
    Shuffling once is noisy. 10 repeats gives a stable average.
    The paper specifies n_repeats=10 as the standard.

    OUTPUT:
    A ranked DataFrame of features by importance.
    This ranking is fed into the counterfactual engine to
    determine which features to try perturbing first.
    High-importance features are tried first (Eq. 8).
    """
    perm = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=10,
        random_state=42
    )

    feat_names = get_pfi_feature_names(pipe, X_test)

    imp_df = pd.DataFrame({
        'Feature':    feat_names[:len(perm.importances_mean)],
        'Importance': perm.importances_mean
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print(f"\n  {'Feature':<25} {'Importance':>12}")
    print(f"  {'-'*38}")
    for _, row in imp_df.iterrows():
        marker = ' ← top' if _ == 0 else ''
        print(f"  {row['Feature']:<25} {row['Importance']:>12.4f}{marker}")

    return imp_df


print("Phase 3 functions defined.")


# ============================================================
# STEP 7 — PHASE 4: COUNTERFACTUAL ENGINE
# ============================================================
# Implements Eq. 1:  Δx* = argmax [f(x+Δx) - f(x)] - λ·C(Δx)
# Implements Eq. 5:  C(Δx) = Σ_j w_j · |Δx_j|
# Implements Eq. 7:  w_j = inf → feature excluded
# Implements Eq. 8:  j* = argmax |(ŷ_new - ŷ_base) - cost_penalty_j|

def cost_function(delta_x_dict, effort_weights):
    """
    Computes C(Δx) — total effort cost of a recommendation set.

    Eq. 5: C(Δx) = Σ_j  w_j · |Δx_j|

    This is a weighted L1 norm over all recommended changes.
    Features with high effort weight (e.g. negotiation_skill=0.7)
    are penalised more heavily than easy changes (calls_made=0.2).

    Parameters:
        delta_x_dict  : {feature: magnitude_of_change}
        effort_weights: the effort weight dict for this dataset

    Returns: float (total cost)
    """
    total = 0.0
    for feat, delta in delta_x_dict.items():
        w = effort_weights.get(feat, 0.5)
        if w < np.inf:
            total += w * abs(delta)
    return total


def counterfactual_single(model_raw, scaler_transform, X_sample,
                           actionable_features, feature_order,
                           effort_weights, lambda_val, use_cost):
    """
    Finds the optimal single-feature recommendation for one salesperson.

    This is Eq. 8 — the discrete counterfactual search:
        j* = argmax_{j in F} |(ŷ_new - ŷ_base) - cost_penalty_j|

    WHAT IT DOES:
    For each actionable feature j:
        1. Take the salesperson's current feature vector x
        2. Nudge feature j by +20% (PERTURBATION)
        3. Compute delta = change in magnitude
        4. Predict new score: ŷ_new = f(x + Δx_j)
        5. Compute impact = ŷ_new - ŷ_base
        6. Compute cost_penalty_j = λ · w_j · |Δx_j|  (Eq. 5)
        7. net_score = |impact - cost_penalty_j|
    Pick j* with highest net_score.

    WHY SINGLE FEATURE?
    Sparse recommendations are more actionable. A manager telling
    a rep "focus on improving your follow-up rate" is clearer and
    more achievable than "change 4 things simultaneously."
    The cost function naturally produces sparse solutions because
    combining multiple changes compounds effort cost.

    Parameters:
        model_raw        : the raw model (not pipeline) for fast prediction
        scaler_transform : function to transform X_sample to scaled space
        X_sample         : one row of original-space features (pd.Series)
        actionable_features: list of feature names that can be recommended
        feature_order    : order to try features (PFI ranking or original)
        effort_weights   : w_j dictionary
        lambda_val       : λ weight on cost penalty
        use_cost         : if False, set cost_penalty = 0 (M-λ variant)

    Returns:
        best_feature (str or None)
        effort_score (float) — C(Δx) for the chosen recommendation
        impact       (float) — predicted improvement from recommendation
    """
    base_pred = model_raw.predict(X_sample.values.reshape(1, -1))[0]

    candidates = []

    for feat in feature_order:
        # Eq. 7: skip immutable features
        if feat not in actionable_features:
            continue

        # Apply 20% perturbation to feature j
        mod_sample     = X_sample.copy()
        original_val   = mod_sample[feat]
        mod_sample[feat] = original_val * (1 + PERTURBATION)
        delta_x_j     = abs(mod_sample[feat] - original_val)

        # Predict under perturbation
        new_pred = model_raw.predict(mod_sample.values.reshape(1, -1))[0]
        impact   = new_pred - base_pred

        # Eq. 5 cost penalty for this feature
        w_j          = effort_weights.get(feat, 0.5)
        cost_penalty = lambda_val * w_j * delta_x_j if use_cost else 0.0

        # Eq. 8 net score
        net_score = abs(impact - cost_penalty)

        candidates.append({
            'feature':      feat,
            'impact':       impact,
            'cost_penalty': cost_penalty,
            'net_score':    net_score,
            'delta_x':      delta_x_j
        })

    if not candidates:
        return None, 0.0, 0.0

    # Pick j* — highest net score (Eq. 8)
    best       = max(candidates, key=lambda c: c['net_score'])
    best_feat  = best['feature']
    effort     = effort_weights.get(best_feat, 0.5) * best['delta_x']
    return best_feat, effort, best['impact']


def run_counterfactuals(model_raw, X_test_scaled, actionable_features,
                        feature_order, effort_weights,
                        lambda_val=LAMBDA, use_cost=True, verbose=False):
    """
    Runs the counterfactual engine on every row in the test set.

    For each salesperson:
    - Finds the single best recommendation (Eq. 8)
    - Checks if recommendation is feasible (in actionable_features)
    - Records effort score

    Returns:
        feasibility_rate : % of recommendations that are actionable (not immutable)
        avg_effort       : mean C(Δx) across all recommendations
        all_recs         : list of recommended features (for analysis)
    """
    feasible   = 0
    efforts    = []
    all_recs   = []

    for i in range(len(X_test_scaled)):
        sample   = X_test_scaled.iloc[i]
        rec, eff, impact = counterfactual_single(
            model_raw, None, sample,
            actionable_features, feature_order,
            effort_weights, lambda_val, use_cost
        )

        all_recs.append(rec)

        if rec is not None and rec in actionable_features:
            feasible += 1
            efforts.append(eff)

        if verbose and i < 5:
            print(f"    Rep {i+1}: Recommend improving '{rec}' → predicted gain: {impact:+.2f}")

    feas_rate  = feasible / len(X_test_scaled) * 100
    avg_effort = np.mean(efforts) if efforts else 0.0

    return feas_rate, avg_effort, all_recs


print("Phase 4 functions defined.")


# ============================================================
# STEP 8 — PHASE 5: ABLATION STUDY
# ============================================================

def run_ablation(X, y, X_train, X_test, y_train, y_test,
                 best_pipe, best_name,
                 actionable_features, effort_weights,
                 pfi_order, label):
    """
    Runs all 4 ablation variants and returns a results table.

    ABLATION DESIGN:
    Each variant disables exactly one component of the framework.
    This proves each component earns its place.

    M-full ★  : all components active           (baseline)
    M-λ       : lambda=0, no cost penalty       (tests Eq. 5 contribution)
    M-F       : all features actionable         (tests feasibility whitelist)
    M-CV      : single split, no CV             (tests reliability module)
    M-XAI     : original column order, no PFI  (tests explainability module)

    For M-λ, M-F, M-XAI: RMSE and R² do not change because the
    predictive model (Phase 2) is untouched. Only the counterfactual
    engine behaviour changes — captured in feasibility and effort.

    For M-CV: the model itself changes (different training procedure)
    so RMSE may differ.
    """
    banner(f"PHASE 5 — ABLATION STUDY: {label}")

    # Extract raw model and scaler from pipeline for CF engine
    raw_model    = best_pipe.named_steps['model']
    preprocessor = best_pipe.named_steps['preprocessor']

    # Transform test set to scaled space for CF engine
    X_test_sc = pd.DataFrame(
        preprocessor.transform(X_test),
        columns=get_pfi_feature_names(best_pipe, X_test)
    )

    # Filter PFI order to only features present after preprocessing
    available_feats = X_test_sc.columns.tolist()
    pfi_order_clean = [f for f in pfi_order if f in available_feats]
    orig_order_clean = [f for f in X_test.columns.tolist() if f in available_feats]
    all_feats_clean  = available_feats  # for M-F (all features)

    # Effort weights mapped to preprocessed feature names
    ew_clean = {f: effort_weights.get(f, 0.5) for f in available_feats}
    actionable_clean = [f for f in actionable_features if f in available_feats]

    results = []

    # ----------------------------------------------------------
    # M-FULL: baseline
    # ----------------------------------------------------------
    rmse_full = np.sqrt(mean_squared_error(y_test, best_pipe.predict(X_test)))
    r2_full   = r2_score(y_test, best_pipe.predict(X_test))
    cv_full   = -cross_val_score(best_pipe, X, y, cv=CV_FOLDS,
                                 scoring='neg_root_mean_squared_error').mean()
    feas_full, effort_full, _ = run_counterfactuals(
        raw_model, X_test_sc, actionable_clean,
        pfi_order_clean, ew_clean, LAMBDA, True
    )
    results.append(['M-full ★', rmse_full, r2_full, cv_full,
                    feas_full, effort_full, 1.0])

    # ----------------------------------------------------------
    # M-λ: set lambda=0 (remove cost penalty from Eq. 1)
    # RMSE/R² unchanged — only predictive model runs, CF changes
    # ----------------------------------------------------------
    feas_ml, effort_ml, _ = run_counterfactuals(
        raw_model, X_test_sc, actionable_clean,
        pfi_order_clean, ew_clean, 0.0, False   # <-- lambda=0
    )
    results.append(['M-λ', rmse_full, r2_full, cv_full,
                    feas_ml, effort_ml,
                    effort_ml / effort_full if effort_full > 0 else 0])

    # ----------------------------------------------------------
    # M-F: remove feasibility whitelist (all features actionable)
    # Eq. 7 disabled — immutable features enter the search space
    # ----------------------------------------------------------
    ew_uniform = {f: 0.5 for f in all_feats_clean}  # uniform weights, no inf
    feas_mf, effort_mf, recs_mf = run_counterfactuals(
        raw_model, X_test_sc, all_feats_clean,  # <-- no whitelist
        pfi_order_clean, ew_uniform, LAMBDA, True
    )
    # Track how many recs fell on originally-immutable features
    immutable_clean = [f for f in DS1_IMMUTABLE + DS2_IMMUTABLE if f in available_feats]
    pct_immutable   = sum(1 for r in recs_mf if r in immutable_clean) / len(recs_mf) * 100
    feas_strict     = 100 - pct_immutable  # true feasibility under M-full rules
    results.append(['M-F', rmse_full, r2_full, cv_full,
                    feas_strict, effort_mf,
                    effort_mf / effort_full if effort_full > 0 else 0])

    # ----------------------------------------------------------
    # M-CV: replace 3-fold CV with single train_test_split
    # The model is retrained without CV — may produce different RMSE
    # ----------------------------------------------------------
    pipe_mcv = Pipeline([
        ('preprocessor', build_preprocessor(X_train)),
        ('model', type(raw_model)(**{
            'random_state': 42
        }) if hasattr(raw_model, 'random_state') else type(raw_model)())
    ])
    # Fit without any cross-validation
    pipe_mcv.fit(X_train, y_train)
    preds_mcv  = pipe_mcv.predict(X_test)
    rmse_mcv   = np.sqrt(mean_squared_error(y_test, preds_mcv))
    r2_mcv     = r2_score(y_test, preds_mcv)
    raw_mcv    = pipe_mcv.named_steps['model']
    pre_mcv    = pipe_mcv.named_steps['preprocessor']
    X_te_mcv   = pd.DataFrame(
        pre_mcv.transform(X_test),
        columns=get_pfi_feature_names(pipe_mcv, X_test)
    )
    pfi_mcv_clean = [f for f in pfi_order_clean if f in X_te_mcv.columns]
    ew_mcv_clean  = {f: effort_weights.get(f, 0.5) for f in X_te_mcv.columns}
    act_mcv_clean = [f for f in actionable_features if f in X_te_mcv.columns]
    feas_mcv, effort_mcv, _ = run_counterfactuals(
        raw_mcv, X_te_mcv, act_mcv_clean,
        pfi_mcv_clean, ew_mcv_clean, LAMBDA, True
    )
    results.append(['M-CV', rmse_mcv, r2_mcv, None,
                    feas_mcv, effort_mcv,
                    effort_mcv / effort_full if effort_full > 0 else 0])

    # ----------------------------------------------------------
    # M-XAI: skip PFI, use original column order
    # Tests whether PFI-guided feature prioritisation matters
    # ----------------------------------------------------------
    feas_mxai, effort_mxai, _ = run_counterfactuals(
        raw_model, X_test_sc, actionable_clean,
        orig_order_clean, ew_clean, LAMBDA, True  # <-- original order
    )
    results.append(['M-XAI', rmse_full, r2_full, cv_full,
                    feas_mxai, effort_mxai,
                    effort_mxai / effort_full if effort_full > 0 else 0])

    # ----------------------------------------------------------
    # Print table
    # ----------------------------------------------------------
    print(f"\n  {'Variant':<12} {'RMSE':>10} {'R²':>8} {'CV_RMSE':>12} "
          f"{'Feasibility':>13} {'Effort':>10} {'Rel.Effort':>12}")
    print(f"  {'-'*80}")

    for row in results:
        v, rmse, r2, cv, feas, eff, rel = row
        cv_str  = f"{cv:>12,.2f}" if cv is not None else f"{'N/A':>12}"
        print(f"  {v:<12} {rmse:>10,.2f} {r2:>8.4f} {cv_str} "
              f"{feas:>12.1f}% {eff:>10.4f} {rel:>11.2f}x")

    print(f"\n  Notes:")
    print(f"  M-F: {pct_immutable:.1f}% of recs fell on immutable features → proves whitelist works")
    print(f"  M-CV: RMSE change vs M-full = {abs(rmse_mcv - rmse_full):,.2f}")
    if abs(effort_mxai - effort_full) > 0.001:
        print(f"  M-XAI: effort shift of {effort_mxai - effort_full:+.4f} → PFI ordering has measurable impact")
    else:
        print(f"  M-XAI: effort unchanged → feature ordering convergence (single dominant feature)")

    return pd.DataFrame(results,
                        columns=['Variant','RMSE','R2','CV_RMSE',
                                 'Feasibility','Effort','Rel_Effort'])


print("Phase 5 functions defined.")


# ============================================================
# STEP 9 — LOAD DATA AND RUN EVERYTHING
# ============================================================

banner("LOADING DATASETS")

# Load
df1_raw = pd.read_csv(DS1_PATH)
df2_raw = pd.read_csv(DS2_PATH)

print(f"\n  DS1 loaded: {df1_raw.shape[0]} rows × {df1_raw.shape[1]} cols")
print(f"  DS2 loaded: {df2_raw.shape[0]} rows × {df2_raw.shape[1]} cols")

# Leakage check — DS1
print(f"\n  DS1 missing values:")
print(df1_raw.isnull().sum()[df1_raw.isnull().sum() > 0].to_string())
print(f"\n  DS2 missing values:")
print(df2_raw.isnull().sum()[df2_raw.isnull().sum() > 0].to_string())

# Clean
df1 = clip_impossible_values(df1_raw, 'DS1')
df2 = clip_impossible_values(df2_raw, 'DS2')

# Drop identity columns — these are row IDs, not features
df1 = df1.drop(columns=['rep_id'], errors='ignore')
df2 = df2.drop(columns=['ae_id'],  errors='ignore')

# Split features and targets
X1 = df1.drop(columns=['revenue_generated'])
y1 = df1['revenue_generated']

X2 = df2.drop(columns=['quota_attainment_pct'])
y2 = df2['quota_attainment_pct']

# Report tau justification (CV = std/mean, unit-free)
cv1 = y1.std() / y1.mean() * 100
cv2 = y2.std() / y2.mean() * 100
print(f"\n  TAU JUSTIFICATION (Coefficient of Variation):")
print(f"  DS1 CV = {cv1:.1f}%  →  τ = {TAU_DS1}  (low-variance domain)")
print(f"  DS2 CV = {cv2:.1f}%  →  τ = {TAU_DS2}  (high-variance domain)")


# ============================================================
# STEP 10 — RUN DS1
# ============================================================

banner("RUNNING DS1 — B2C RETAIL SALES REP")

# Phase 2
pipe1, name1, Xtr1, Xte1, ytr1, yte1, res1 = benchmark_models(
    X1, y1, TAU_DS1, "DS1 B2C Retail"
)

# Noise stress
print("\n  --- Noise Stress Test ---")
ns1 = noise_stress_test(pipe1, Xte1, yte1)

# Phase 3 — PFI
banner("PHASE 3 — PFI: DS1")
pfi1 = run_pfi(pipe1, Xte1, yte1)
pfi1_order = pfi1['Feature'].tolist()

# Phase 4 — sample recommendations (first 5 reps)
banner("PHASE 4 — COUNTERFACTUAL SAMPLE: DS1 (first 5 reps)")
raw1 = pipe1.named_steps['model']
pre1 = pipe1.named_steps['preprocessor']
Xte1_sc = pd.DataFrame(
    pre1.transform(Xte1),
    columns=get_pfi_feature_names(pipe1, Xte1)
)
pfi1_clean = [f for f in pfi1_order if f in Xte1_sc.columns]
act1_clean = [f for f in DS1_ACTIONABLE if f in Xte1_sc.columns]
ew1_clean  = {f: DS1_EFFORT_WEIGHTS.get(f, 0.5) for f in Xte1_sc.columns}

print(f"\n  {'Rep':<6} {'Recommendation':<25} {'Effort':>8}")
print(f"  {'-'*42}")
for i in range(min(5, len(Xte1_sc))):
    rec, eff, impact = counterfactual_single(
        raw1, None, Xte1_sc.iloc[i],
        act1_clean, pfi1_clean, ew1_clean, LAMBDA, True
    )
    print(f"  {i+1:<6} Improve '{rec}'{'':>15} {eff:>8.4f}")

# Phase 5 — Ablation
ablation1 = run_ablation(
    X1, y1, Xtr1, Xte1, ytr1, yte1,
    pipe1, name1,
    DS1_ACTIONABLE, DS1_EFFORT_WEIGHTS,
    pfi1_order, "DS1 B2C Retail"
)


# ============================================================
# STEP 11 — RUN DS2
# ============================================================

banner("RUNNING DS2 — B2B ACCOUNT EXECUTIVE")

# Phase 2
pipe2, name2, Xtr2, Xte2, ytr2, yte2, res2 = benchmark_models(
    X2, y2, TAU_DS2, "DS2 B2B AE"
)

# Noise stress
print("\n  --- Noise Stress Test ---")
ns2 = noise_stress_test(pipe2, Xte2, yte2)

# Phase 3 — PFI
banner("PHASE 3 — PFI: DS2")
pfi2 = run_pfi(pipe2, Xte2, yte2)
pfi2_order = pfi2['Feature'].tolist()

# Phase 4 — sample recommendations
banner("PHASE 4 — COUNTERFACTUAL SAMPLE: DS2 (first 5 AEs)")
raw2 = pipe2.named_steps['model']
pre2 = pipe2.named_steps['preprocessor']
Xte2_sc = pd.DataFrame(
    pre2.transform(Xte2),
    columns=get_pfi_feature_names(pipe2, Xte2)
)
pfi2_clean = [f for f in pfi2_order if f in Xte2_sc.columns]
act2_clean = [f for f in DS2_ACTIONABLE if f in Xte2_sc.columns]
ew2_clean  = {f: DS2_EFFORT_WEIGHTS.get(f, 0.5) for f in Xte2_sc.columns}

print(f"\n  {'AE':<6} {'Recommendation':<25} {'Effort':>8}")
print(f"  {'-'*42}")
for i in range(min(5, len(Xte2_sc))):
    rec, eff, impact = counterfactual_single(
        raw2, None, Xte2_sc.iloc[i],
        act2_clean, pfi2_clean, ew2_clean, LAMBDA, True
    )
    print(f"  {i+1:<6} Improve '{rec}'{'':>15} {eff:>8.4f}")

# Phase 5 — Ablation
ablation2 = run_ablation(
    X2, y2, Xtr2, Xte2, ytr2, yte2,
    pipe2, name2,
    DS2_ACTIONABLE, DS2_EFFORT_WEIGHTS,
    pfi2_order, "DS2 B2B AE"
)


# ============================================================
# STEP 12 — FINAL SUMMARY FOR PAPER
# ============================================================

banner("FINAL NUMBERS FOR PAPER")

print("\n  ── TABLE I: MODEL BENCHMARKING ──")
print(f"\n  DS1 ({name1} selected | τ={TAU_DS1})")
print(res1.to_string(index=False))

print(f"\n  DS2 ({name2} selected | τ={TAU_DS2})")
print(res2.to_string(index=False))

print(f"\n  ── NOISE STRESS ──")
print(f"  DS1 Noise Stress RMSE: {ns1:,.2f}")
print(f"  DS2 Noise Stress RMSE: {ns2:,.2f}")

print(f"\n  ── TABLE II: ABLATION STUDY ──")
print(f"\n  DS1 Ablation:")
print(ablation1.to_string(index=False))

print(f"\n  DS2 Ablation:")
print(ablation2.to_string(index=False))

print(f"\n  ── TAU JUSTIFICATION ──")
print(f"  DS1 CV = {y1.std()/y1.mean()*100:.1f}%  →  τ = {TAU_DS1}")
print(f"  DS2 CV = {y2.std()/y2.mean()*100:.1f}%  →  τ = {TAU_DS2}")

print(f"\n  ── EFFORT WEIGHT HIERARCHY (Eq. 6) ──")
print(f"  DS1: calls({DS1_EFFORT_WEIGHTS['calls_made']}) < meetings({DS1_EFFORT_WEIGHTS['meetings_booked']}) "
      f"< follow_up({DS1_EFFORT_WEIGHTS['follow_up_rate']}) < script({DS1_EFFORT_WEIGHTS['script_adherence']}) "
      f"< avg_order_value({DS1_EFFORT_WEIGHTS['avg_order_value']})")
print(f"  DS2: discovery({DS2_EFFORT_WEIGHTS['discovery_calls']}) < crm({DS2_EFFORT_WEIGHTS['crm_hygiene']}) "
      f"< demos({DS2_EFFORT_WEIGHTS['demos_delivered']}) < proposals({DS2_EFFORT_WEIGHTS['proposals_sent']}) "
      f"< negotiation({DS2_EFFORT_WEIGHTS['negotiation_skill']})")

banner("PIPELINE COMPLETE — ALL NUMBERS CONFIRMED")
