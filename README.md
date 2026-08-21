# Digital Twin- Pump Health Monitor

## Live Demo
[Click here to view dashboard](https://digital-twin-pump-simulation.onrender.com)

## Problem
Unexpected pump failures cause costly downtime in industrial facilities.
This digital twin monitors pump health in real-time and detects faults
before they cause failure.

## What is a Digital Twin?
A virtual copy of a physical pump that runs in parallel.
When sensor behavior diverges from normal, the twin detects it first.

## Fault Types Detected
- Bearing Wear - vibration + temperature increase
- Cavitation - flow drop + vibration spikes  
- Blockage - flow drop + pressure buildup

## Approach
- Physics-based pump simulation (pump curve, system curve)
- Synthetic sensor data generation for 3 fault types
- Random Forest classifier with 99.6% accuracy
- Real-time health score (0-100%)
- Interactive Plotly Dash dashboard

## Tech Stack
Python, PyTorch, Random Forest, Plotly Dash, Render

## Author
Smrity Mallik | MSc AI & Data Science | Mechanical Engineering
