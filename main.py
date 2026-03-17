"""
main.py - Multi-Currency FX Fair Value Platform
================================================
Flask backend serving a quantitative FX dashboard.
Covers: EURUSD, GBPUSD, USDJPY, USDZAR, USDCAD

Run:  pip install flask numpy scipy scikit-learn
      python main.py
Then: open http://localhost:5000
"""

import os
import json
import numpy as np
from datetime import datetime
from flask import Flask, send_file, jsonify
from scipy import stats, optimize, linalg
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ================================================================
# SIMULATION ENGINE
# ================================================================

PAIRS = {
    "EURUSD": {"base": "EUR", "quote": "USD", "start": 1.15, "mu": 1.15, "vol": 0.07, "theta": 0.03},
    "GBPUSD": {"base": "GBP", "quote": "USD", "start": 1.55, "mu": 1.40, "vol": 0.08, "theta": 0.02},
    "USDJPY": {"base": "USD", "quote": "JPY", "start": 110.0, "mu": 115.0, "vol": 8.0, "theta": 0.02},
    "USDZAR": {"base": "USD", "quote": "ZAR", "start": 12.0, "mu": 15.0, "vol": 2.0, "theta": 0.01},
    "USDCAD": {"base": "USD", "quote": "CAD", "start": 1.25, "mu": 1.30, "vol": 0.06, "theta": 0.03},
}

def ou_process(mu, sigma, theta, x0, n, rng, dt=1.0/12.0):
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu - x[t-1]) * dt + sigma * np.sqrt(dt) * rng.randn()
    return x

def generate_pair_data(pair_name, seed=None):
    cfg = PAIRS[pair_name]
    rng = np.random.RandomState(seed or hash(pair_name) % 2**31)
    T = 252
    dates = []
    d = datetime(2004, 1, 1)
    for i in range(T):
        dates.append(d.strftime("%Y-%m-%d"))
        m = d.month + 1
        y = d.year
        if m > 12:
            m = 1
            y += 1
        d = datetime(y, m, 1)

    # Macro factors
    rate_diff = ou_process(1.0, 0.5, 0.1, 0.5, T, rng)
    infl_diff = ou_process(0.5, 0.4, 0.12, 0.3, T, rng)
    growth_diff = ou_process(0.0, 0.8, 0.2, 0.2, T, rng)
    ca_diff = ou_process(-2.0, 1.5, 0.05, -1.5, T, rng)
    vix = np.clip(ou_process(18.0, 4.0, 0.15, 16.0, T, rng), 9, 80)
    risk_appetite = ou_process(0.0, 0.3, 0.1, 0.0, T, rng)
    real_rate_diff = rate_diff - infl_diff
    slope_diff = ou_process(0.3, 0.3, 0.1, 0.2, T, rng)

    # Spot via OU + fundamentals
    spot = np.zeros(T)
    spot[0] = cfg["start"]
    base_level = cfg["mu"]
    scale = cfg["vol"] / 12.0
    for t in range(1, T):
        pull = cfg["theta"] * (cfg["mu"] - spot[t-1]) / 12.0
        macro = -0.01 * rate_diff[t] + 0.005 * ca_diff[t] / 10.0 - 0.003 * (vix[t] - 18) / 10.0
        macro = macro * scale * 3
        spot[t] = spot[t-1] + pull + macro + scale * rng.randn()
    # Ensure positive and realistic range
    low_clip = base_level * 0.5
    high_clip = base_level * 1.8
    spot = np.clip(spot, low_clip, high_clip)

    features = {
        "rate_diff": rate_diff.tolist(),
        "infl_diff": infl_diff.tolist(),
        "growth_diff": growth_diff.tolist(),
        "ca_diff": ca_diff.tolist(),
        "vix": vix.tolist(),
        "risk_appetite": risk_appetite.tolist(),
        "real_rate_diff": real_rate_diff.tolist(),
        "slope_diff": slope_diff.tolist(),
    }
    return dates, spot, features

# ================================================================
# ECONOMETRIC ENGINE
# ================================================================

def run_ols(y, X):
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])
    try:
        beta = linalg.lstsq(Xc, y)[0]
    except Exception:
        beta = np.zeros(Xc.shape[1])
    fitted = Xc @ beta
    resid = y - fitted
    ss_tot = float(np.sum((y - np.mean(y))**2))
    ss_res = float(np.sum(resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {"name": "OLS", "fitted": fitted, "residuals": resid, "r2": r2, "rmse": rmse, "beta": beta}

def run_ridge(y, X, alpha=1.0):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=alpha).fit(Xs, y)
    fitted = model.predict(Xs)
    resid = y - fitted
    ss_tot = float(np.sum((y - np.mean(y))**2))
    ss_res = float(np.sum(resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {"name": "Ridge", "fitted": fitted, "residuals": resid, "r2": r2, "rmse": rmse, "coef": model.coef_}

def run_lasso(y, X, alpha=0.01):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Lasso(alpha=alpha, max_iter=50000).fit(Xs, y)
    fitted = model.predict(Xs)
    resid = y - fitted
    ss_tot = float(np.sum((y - np.mean(y))**2))
    ss_res = float(np.sum(resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resid**2)))
    n_sel = int(np.sum(np.abs(model.coef_) > 1e-6))
    return {"name": "Lasso", "fitted": fitted, "residuals": resid, "r2": r2, "rmse": rmse, "coef": model.coef_, "n_selected": n_sel}

def run_rolling(y, X, window=36):
    T = len(y)
    fitted = np.full(T, np.nan)
    betas = np.full((T, X.shape[1]), np.nan)
    for t in range(window, T):
        Xw = np.column_stack([np.ones(window), X[t-window:t]])
        yw = y[t-window:t]
        try:
            b = linalg.lstsq(Xw, yw)[0]
            xt = np.concatenate([[1.0], X[t]])
            fitted[t] = float(xt @ b)
            betas[t] = b[1:]
        except Exception:
            pass
    valid = ~np.isnan(fitted)
    resid = np.where(valid, y - fitted, np.nan)
    rmse = float(np.sqrt(np.nanmean(resid[valid]**2))) if valid.any() else 99.0
    return {"name": "Rolling", "fitted": fitted, "residuals": resid, "rmse": rmse, "betas": betas}

def run_bayesian(y, X, prior_precision=0.1):
    n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])
    kf = k + 1
    L0 = prior_precision * np.eye(kf)
    b0 = np.zeros(kf)
    Ln = L0 + Xc.T @ Xc
    try:
        Ln_inv = linalg.inv(Ln)
    except Exception:
        Ln_inv = linalg.pinv(Ln)
    bn = Ln_inv @ (L0 @ b0 + Xc.T @ y)
    fitted = Xc @ bn
    resid = y - fitted
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {"name": "Bayesian", "fitted": fitted, "residuals": resid, "rmse": rmse}

def run_garch(residuals):
    r = residuals[~np.isnan(residuals)]
    T = len(r)
    if T < 20:
        return {"omega": 0.0, "alpha": 0.0, "beta": 0.0, "cond_vol": []}
    mu = float(np.mean(r))
    e = r - mu
    var_e = float(np.var(e))
    def neg_ll(params):
        om, al, be = params
        if om < 1e-8 or al < 0 or be < 0 or al + be >= 1:
            return 1e10
        s2 = np.zeros(T)
        s2[0] = var_e
        for t in range(1, T):
            s2[t] = om + al * e[t-1]**2 + be * s2[t-1]
            if s2[t] < 1e-10:
                s2[t] = 1e-10
        return 0.5 * float(np.sum(np.log(s2) + e**2 / s2))
    try:
        res = optimize.minimize(neg_ll, [var_e*0.05, 0.1, 0.85], method="Nelder-Mead", options={"maxiter": 3000})
        om, al, be = res.x
    except Exception:
        om, al, be = var_e*0.05, 0.1, 0.85
    s2 = np.zeros(T)
    s2[0] = var_e
    for t in range(1, T):
        s2[t] = om + al * e[t-1]**2 + be * s2[t-1]
    return {"omega": float(om), "alpha": float(al), "beta": float(be),
            "persistence": float(al + be), "cond_vol": np.sqrt(s2).tolist()}

def run_monte_carlo(current_spot, fair_value, vol, n_sims=5000, horizon=12):
    paths = np.zeros((n_sims, horizon + 1))
    paths[:, 0] = current_spot
    monthly_vol = vol / np.sqrt(12)
    for t in range(1, horizon + 1):
        gap = fair_value - paths[:, t-1]
        reversion = 0.05 * gap
        noise = monthly_vol * np.random.standard_t(df=5, size=n_sims)
        paths[:, t] = paths[:, t-1] + reversion + noise
    pcts = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        pcts["p" + str(p)] = np.percentile(paths, p, axis=0).tolist()
    return pcts

def compute_ensemble(models, y):
    inv_rmse = {}
    for m in models:
        r = m["residuals"]
        valid = ~np.isnan(r)
        if valid.sum() > 0:
            rmse = float(np.sqrt(np.mean(r[valid]**2)))
            inv_rmse[m["name"]] = 1.0 / (rmse + 1e-10)
    total = sum(inv_rmse.values())
    weights = {k: round(v / total, 4) for k, v in inv_rmse.items()}
    ensemble = np.zeros(len(y))
    tw = 0.0
    for m in models:
        w = weights.get(m["name"], 0.0)
        f = m["fitted"]
        valid = ~np.isnan(f)
        ensemble[valid] += w * f[valid]
        tw += w
    ensemble /= tw
    resid = y - ensemble
    return ensemble, weights, resid

def analyze_misalignment(residuals):
    sigma = float(np.std(residuals))
    extreme_2s = int(np.sum(np.abs(residuals) > 2 * sigma))
    extreme_3s = int(np.sum(np.abs(residuals) > 3 * sigma))
    pct_2s = round(extreme_2s / len(residuals) * 100, 1)
    # Mean reversion at 6m
    idx = np.where(np.abs(residuals) > 2 * sigma)[0]
    rev_6m = 0
    total = 0
    for i in idx:
        if i + 6 < len(residuals):
            if abs(residuals[i + 6]) < abs(residuals[i]):
                rev_6m += 1
            total += 1
    rev_pct = round(rev_6m / total * 100, 0) if total > 0 else 0
    return {"sigma": round(sigma, 6), "extreme_2s": extreme_2s, "pct_2s": pct_2s,
            "extreme_3s": extreme_3s, "mean_rev_6m": rev_pct}

# ================================================================
# FULL PAIR ANALYSIS
# ================================================================

def analyze_pair(pair_name):
    dates, spot, features = generate_pair_data(pair_name)
    y = spot
    feat_names = list(features.keys())
    X = np.column_stack([np.array(features[f]) for f in feat_names])

    # Fill NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    y_clean = y[mask]
    X_clean = X[mask]
    dates_clean = [dates[i] for i in range(len(dates)) if mask[i]]

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_clean)

    # Run models
    ols = run_ols(y_clean, X_std)
    ridge = run_ridge(y_clean, X_clean)
    lasso = run_lasso(y_clean, X_clean)
    rolling = run_rolling(y_clean, X_std, window=36)
    bayesian = run_bayesian(y_clean, X_std)

    models = [ols, ridge, lasso, rolling, bayesian]
    ensemble_fv, weights, ens_resid = compute_ensemble(models, y_clean)

    # GARCH
    garch = run_garch(ens_resid)

    # Monte Carlo
    vol = float(np.std(ens_resid)) * np.sqrt(12)
    fan = run_monte_carlo(y_clean[-1], ensemble_fv[-1], vol)

    # Misalignment
    mis = analyze_misalignment(ens_resid)

    # Z-score rolling
    r_series = ens_resid
    z_scores = []
    for t in range(len(r_series)):
        start = max(0, t - 60)
        window = r_series[start:t+1]
        if len(window) > 12:
            mu = float(np.mean(window))
            sig = float(np.std(window))
            z_scores.append(round((r_series[t] - mu) / (sig + 1e-10), 4))
        else:
            z_scores.append(0.0)

    # Current state
    current_spot = round(float(y_clean[-1]), 4)
    current_fv = round(float(ensemble_fv[-1]), 4)
    misalign_pct = round((current_spot - current_fv) / current_fv * 100, 2)

    # Model results for table
    model_table = []
    for m in models:
        row = {"name": m["name"], "rmse": round(m["rmse"], 6),
               "r2": round(m.get("r2", 0.0), 4),
               "weight": weights.get(m["name"], 0.0)}
        model_table.append(row)

    # Coefficient importance (from OLS)
    coef_importance = []
    for i, name in enumerate(feat_names):
        coef_importance.append({"name": name, "ols": round(float(ols["beta"][i+1]), 4),
                                "lasso": round(float(lasso["coef"][i]), 4) if i < len(lasso["coef"]) else 0.0})

    # Rolling betas (top 4)
    roll_betas = rolling["betas"]
    avg_imp = np.nanmean(np.abs(roll_betas), axis=0)
    top4 = np.argsort(avg_imp)[-4:]
    rolling_coefs = {}
    for idx in top4:
        if idx < len(feat_names):
            vals = roll_betas[:, idx]
            rolling_coefs[feat_names[idx]] = [round(float(v), 4) if not np.isnan(v) else None for v in vals]

    return {
        "pair": pair_name,
        "base": PAIRS[pair_name]["base"],
        "quote": PAIRS[pair_name]["quote"],
        "dates": dates_clean,
        "spot": [round(float(v), 4) for v in y_clean],
        "fair_value": [round(float(v), 4) for v in ensemble_fv],
        "residuals": [round(float(v), 6) for v in ens_resid],
        "z_scores": z_scores,
        "model_fitted": {m["name"]: [round(float(v), 4) if not np.isnan(v) else None for v in m["fitted"]] for m in models},
        "model_table": model_table,
        "weights": weights,
        "coef_importance": coef_importance,
        "rolling_coefs": rolling_coefs,
        "garch": {"omega": round(garch["omega"], 6), "alpha": round(garch["alpha"], 4),
                  "beta": round(garch["beta"], 4), "persistence": round(garch["persistence"], 4),
                  "cond_vol": [round(v, 6) for v in garch["cond_vol"]]},
        "fan_chart": fan,
        "misalignment": mis,
        "current": {"spot": current_spot, "fair_value": current_fv, "misalign_pct": misalign_pct,
                     "direction": "overvalued" if misalign_pct > 0 else "undervalued"},
        "risk": {"var_95": round(abs(float(np.percentile(np.diff(y_clean) / y_clean[:-1], 5))), 4) if len(y_clean) > 20 else 0,
                 "es_95": round(abs(float(np.mean(np.sort(np.diff(y_clean) / y_clean[:-1])[:max(1, int(len(y_clean)*0.05))]))), 4) if len(y_clean) > 20 else 0},
    }

# ================================================================
# PRECOMPUTE ALL PAIRS
# ================================================================

print("[ENGINE] Computing fair values for all pairs...")
RESULTS = {}
for pair in PAIRS:
    print("  -> {}...".format(pair))
    RESULTS[pair] = analyze_pair(pair)
print("[ENGINE] Done. Starting server...")

# ================================================================
# FLASK ROUTES
# ================================================================

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/api/pairs")
def api_pairs():
    summary = []
    for name, r in RESULTS.items():
        summary.append({
            "pair": name, "base": r["base"], "quote": r["quote"],
            "spot": r["current"]["spot"], "fair_value": r["current"]["fair_value"],
            "misalign_pct": r["current"]["misalign_pct"], "direction": r["current"]["direction"],
        })
    return jsonify(summary)

@app.route("/api/pair/<name>")
def api_pair(name):
    name = name.upper()
    if name not in RESULTS:
        return jsonify({"error": "Pair not found"}), 404
    return jsonify(RESULTS[name])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_encoder = NumpyEncoder

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  FX Fair Value Platform")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)