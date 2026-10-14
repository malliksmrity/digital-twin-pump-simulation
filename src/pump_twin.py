"""Synthetic pump data and an honest fault-detection test.

The notebook generates one 1000-step run per condition (normal, bearing wear,
cavitation, blockage) and reports 99.6 percent accuracy for a random forest.
That number comes from a random row split after the faulty rows were copied
(upsampled) to match the healthy ones, so the same rows sit in train and test.

This script
  1. reproduces the 99.60 percent (label: "notebook"),
  2. generates many independent runs and tests on runs the model has never seen,
  3. compares against two simple baselines and measures how many steps after
     the fault starts the model notices it.

Run from the repo root:  python src/pump_twin.py
Writes reports/results.json and docs/data.js.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ["flow_rate", "pressure", "vibration", "temperature", "power"]
FAULTS = ["normal", "bearing_wear", "cavitation", "blockage"]
FAULT_START = 700
STEPS = 1000


def generate(fault, rng, steps=STEPS, start=FAULT_START):
    """Same physics-flavoured recipe as the notebook. `rng` is a numpy Generator
    or the np.random module (the notebook uses the global state)."""
    t = np.arange(steps)
    flow = 0.065 + rng.normal(0, 0.002, steps)
    pres = 35 + rng.normal(0, 0.5, steps)
    vib = 0.5 + rng.normal(0, 0.05, steps)
    temp = 25 + rng.normal(0, 0.3, steps)
    pwr = 15 + rng.normal(0, 0.2, steps)
    m = t >= start
    n = int(m.sum())
    if fault == "bearing_wear":
        vib[m] += np.linspace(0, 3, n)
        temp[m] += np.linspace(0, 15, n)
    elif fault == "cavitation":
        flow[m] -= np.linspace(0, 0.02, n)
        vib[m] += rng.normal(2, 0.5, n)
    elif fault == "blockage":
        flow[m] -= np.linspace(0, 0.03, n)
        pres[m] += np.linspace(0, 10, n)
        pwr[m] += np.linspace(0, 5, n)
    health = np.ones(steps)
    if fault != "normal":
        health[start:] = 0
    return pd.DataFrame({"time": t, "flow_rate": flow, "pressure": pres, "vibration": vib,
                         "temperature": temp, "power": pwr, "health": health, "fault_type": fault})


def reproduce_notebook():
    """Random forest as in the notebook: seed 42, upsample faulty rows, random 80/20 split."""
    np.random.seed(42)
    data = pd.concat([generate(f, np.random) for f in FAULTS], ignore_index=True)
    healthy, faulty = data[data.health == 1], data[data.health == 0]
    up = resample(faulty, replace=True, n_samples=len(healthy), random_state=42)
    bal = pd.concat([healthy, up]).sample(frac=1, random_state=42)
    X = StandardScaler().fit_transform(bal[SENSORS].values)
    Xtr, Xte, ytr, yte = train_test_split(X, bal["health"].values, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    # how many test rows are exact copies of a training row?
    tr_rows = {tuple(r) for r in np.round(Xtr, 8)}
    dup = float(np.mean([tuple(r) in tr_rows for r in np.round(Xte, 8)]))
    return float(accuracy_score(yte, rf.predict(Xte))), dup


def add_rolling(df, w=10):
    """Causal rolling mean and std inside one run (only looks at the past)."""
    out = df.copy()
    for c in SENSORS:
        out[c + "_m"] = out[c].rolling(w, min_periods=1).mean()
        out[c + "_s"] = out[c].rolling(w, min_periods=2).std().fillna(0)
    return out


def detection_delay(pred_faulty, start=FAULT_START, k=5):
    """Steps after the fault start until the first run of k consecutive 'faulty' predictions."""
    run = 0
    for i in range(start, len(pred_faulty)):
        run = run + 1 if pred_faulty[i] else 0
        if run == k:
            return i - k + 1 - start
    return None


def main():
    nb_acc, dup = reproduce_notebook()

    # independent runs: 20 per condition, seeds 1000.., 14 for training and 6 for testing
    runs = []
    for fi, f in enumerate(FAULTS):
        for r in range(20):
            rng = np.random.default_rng(1000 + fi * 100 + r)
            d = add_rolling(generate(f, rng))
            d["run"] = f"{f}_{r}"
            d["split"] = "train" if r < 14 else "test"
            runs.append(d)
    all_runs = pd.concat(runs, ignore_index=True)
    tr, te = all_runs[all_runs.split == "train"], all_runs[all_runs.split == "test"]
    rolled = SENSORS + [c + s for c in SENSORS for s in ("_m", "_s")]
    y_tr, y_te = tr["health"].values, te["health"].values

    def row(name, pred):
        faulty = y_te == 0
        return {"model": name, "accuracy": round(float(accuracy_score(y_te, pred)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y_te, pred)), 4),
                "recall_faulty": round(float(np.mean(pred[faulty] == 0)), 4),
                "false_alarm_rate": round(float(np.mean(pred[~faulty] == 0)), 4)}

    results = [row("Always predict healthy", np.ones(len(y_te)))]
    # threshold rule: any sensor beyond 4 standard deviations of healthy training data
    hl = tr[tr.health == 1]
    mu, sd = hl[SENSORS].mean(), hl[SENSORS].std()
    z = ((te[SENSORS] - mu) / sd).abs().max(axis=1)
    results.append(row("Simple threshold (any sensor beyond 4 sigma)", np.where(z > 4, 0.0, 1.0)))
    models = {}
    for name, cols in [("Random forest, raw sensors", SENSORS), ("Random forest, with rolling features", rolled)]:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=1)
        rf.fit(tr[cols], y_tr)
        p = rf.predict(te[cols])
        models[name] = (rf, cols, p)
        results.append(row(name, p))

    # detection delay per fault type for the rolling-feature model
    rf, cols, p = models["Random forest, with rolling features"]
    te = te.assign(pred=p)
    delays, missed = {}, {}
    for f in FAULTS[1:]:
        ds = []
        for _, g in te[te.fault_type == f].groupby("run"):
            ds.append(detection_delay(g["pred"].values == 0))
        found = [d for d in ds if d is not None]
        delays[f] = {"median_delay_steps": float(np.median(found)) if found else None,
                     "runs_detected": len(found), "runs_total": len(ds)}

    # one example run for the page
    ex = te[te.run == "bearing_wear_14"]
    example = {"time": ex["time"].tolist()[::5], "vibration": ex["vibration"].round(3).tolist()[::5],
               "temperature": ex["temperature"].round(2).tolist()[::5],
               "pred_faulty": [int(v == 0) for v in ex["pred"]][::5], "fault_start": FAULT_START}
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False).head(6)

    out = {"notebook_replication": {"accuracy": round(nb_acc, 4), "test_rows_identical_to_a_train_row": round(dup, 3)},
           "heldout_runs": {"train_runs": int(tr.run.nunique()), "test_runs": int(te.run.nunique()),
                            "test_rows": int(len(te)), "results": results},
           "detection_delay": delays}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    page = dict(out, example=example, importance=[{"feature": k, "value": round(float(v), 3)} for k, v in imp.items()])
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "data.js").write_text("window.PAGE_DATA = " + json.dumps(page) + ";\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
