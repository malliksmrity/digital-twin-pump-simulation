# Pump fault detection with a simulated digital twin

**In one sentence:** a small simulation of a water pump produces sensor data with three kinds of faults, and a classifier learns to tell a healthy pump from a faulty one.

Project page (charts, plain language): `docs/index.html`, served with GitHub Pages once Pages is enabled for the `/docs` folder.

## The question

Can sensor patterns (flow, pressure, vibration, temperature, power) reveal a developing pump fault early enough to act, and how much does a machine-learning model add over a simple rule?

## Who could benefit, and how

Maintenance engineers in plants with many pumps could use condition monitoring like this to catch faults before a breakdown. This project only shows the idea on simulated data. It has not been tested on a real pump.

## Data and licence

No external data. `src/pump_twin.py` generates the data with fixed seeds: 1000 time steps per run, a fault starts at step 700. Faults: bearing wear (vibration and temperature rise), cavitation (flow drops, vibration jumps), blockage (flow drops, pressure and power rise).

## How it works

1. Simulate a healthy pump and three faulty ones, using a pump curve and added sensor noise (the notebook shows the physics part).
2. Build features: the five sensors plus the average and spread over the last 10 steps.
3. Train a random forest on 56 simulated runs (14 per condition).
4. Test on 24 other runs and measure accuracy, false alarms and how many steps after the fault start it is noticed.

## Results

From `python src/pump_twin.py`, saved in `reports/results.json`.

| What was tested | Accuracy |
|---|---|
| Original notebook: random forest, random row split after copying faulty rows | 99.60 % |
| Same set-up, share of test rows that are exact copies of a training row | 46.6 % |
| Always predict "healthy" (24 held-out runs, 77.5 % of readings are healthy) | 77.50 % |
| Simple rule: any sensor more than 4 standard deviations from healthy | 98.36 % |
| Random forest, raw sensors, held-out runs | 98.78 % |
| Random forest, with rolling features, held-out runs | **99.29 %** |

For the rolling-feature model: 96.9 % of faulty readings are caught, and 0.02 % of healthy readings raise a false alarm. Median delay until five faulty calls in a row: bearing wear 9.5 steps, cavitation 0 steps, blockage 17.5 steps (6 test runs each, all detected).

The notebook's LSTM experiments reached 65.66 %, 62.41 % and 50.08 % accuracy and were dropped. They are kept in the notebook as they were and not re-run.

## Limitations

- Everything is simulated, and the faults are injected as clean shifts. That is why a simple rule almost matches the random forest.
- The old 99.6 % is not wrong arithmetic, but it was measured on rows that also appear in the training data.
- The label switches at step 700 while the physical fault grows slowly from zero, so the first steps after onset look healthy. Accuracy can not reach 100 % honestly.
- The earlier README promised "real-time" monitoring. The project simulates it; it does not connect to a pump.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/pump_twin.py         # writes reports/results.json and docs/data.js
# for the notebook (plots, LSTM, needs PyTorch):
pip install -r requirements-notebook.txt
```

## What is coursework and what I extended

This is a personal project, not coursework. Kept as written: the notebook (simulation, LSTM attempts, random forest). Extended in this cleanup: the seeded script with held-out runs, baselines and detection delay, plus the static page. The Plotly Dash app for Render was moved to `_archive/` because the project now uses static pages only.

## For technical readers

- Held-out protocol: 20 runs per condition, seeds 1000 and up. Runs 0 to 13 train, 14 to 19 test, so no run is split across train and test.
- Rolling features are causal (only past values), computed inside each run.
- Random forest: 100 trees, `class_weight="balanced"`, `random_state=42`. No upsampling.
- Replication of the notebook: global `np.random.seed(42)`, same generation order, same upsampling and split. Gives 99.60 %.
- Versions pinned in `requirements.txt` (Python 3.11).

## Repository layout

```
notebooks/   digital_twin.ipynb (original)
src/         pump_twin.py
reports/     results.json
docs/        static project page
_archive/    moved files: Dash app, Render files, notebook checkpoint, original README
```

Smrity Mallik, MSc AI and Data Science (Gisma, Berlin), B.Eng. Mechanical Engineering.
GitHub: github.com/malliksmrity. LinkedIn: linkedin.com/in/smritymallik.
