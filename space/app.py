"""Pump fault detection on a simulated pump: pick a fault and a seed, see what the random forest says."""
import matplotlib
matplotlib.use("Agg")
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pump_twin import FAULTS, FAULT_START, SENSORS, add_rolling, detection_delay, generate

runs = []
for fi, f in enumerate(FAULTS):
    for r in range(14):
        runs.append(add_rolling(generate(f, np.random.default_rng(1000 + fi * 100 + r))))
train = pd.concat(runs)
COLS = SENSORS + [c + s for c in SENSORS for s in ("_m", "_s")]
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=1).fit(train[COLS], train["health"])


def run(fault, seed):
    d = add_rolling(generate(fault, np.random.default_rng(int(seed) + 50000)))
    pred_faulty = rf.predict(d[COLS]) == 0
    delay = detection_delay(pred_faulty) if fault != "normal" else None
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(d["time"], d["vibration"], color="#0b5d6b")
    ax[0].set_ylabel("Vibration (mm/s)"); ax[0].set_xlabel("Time step")
    ax[0].fill_between(d["time"], 0, d["vibration"].max(), where=pred_faulty, color="#0b5d6b", alpha=0.15)
    ax[0].axvline(FAULT_START, color="black", linestyle=":")
    ax[1].plot(d["time"], d["temperature"], color="#b35c1e")
    ax[1].set_ylabel("Temperature (C)"); ax[1].set_xlabel("Time step"); ax[1].axvline(FAULT_START, color="black", linestyle=":")
    fig.tight_layout()
    if fault == "normal":
        msg = f"Normal run: {int(pred_faulty.sum())} of {len(d)} steps flagged as faulty (false alarms)."
    elif delay is not None:
        msg = f"Fault starts at step {FAULT_START}. Detected {delay} steps later."
    else:
        msg = "Fault not detected."
    return msg + " Simulated data. On 24 unseen runs the model was right for 99.3 percent of readings; a simple threshold rule got 98.4.", fig


demo = gr.Interface(run, [gr.Dropdown(FAULTS, value="bearing_wear", label="Condition"), gr.Slider(0, 99, value=1, step=1, label="Random seed")],
                    [gr.Textbox(label="Result"), gr.Plot()], title="Pump fault detection (simulated)",
                    description="Code: github.com/malliksmrity/digital-twin-pump-simulation")
if __name__ == "__main__":
    demo.launch()
