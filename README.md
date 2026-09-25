# Chip Power Anomaly Detector

A machine learning prototype that detects abnormal power consumption patterns in a simulated chip, comparing a simple statistical baseline (Z-score) against an ML model (Isolation Forest).

Built as a technical evaluation project for the Asymmetric Club, CIT Chennai.

## What This Project Does

Real chips draw power in predictable patterns when healthy. Faults, like short circuits, overheating, or dead components, often show up as unusual changes in power consumption before they're caught any other way.

This project:
1. Simulates 2000 power readings from a healthy chip (~50mW baseline)
2. Injects 3 realistic fault types
3. Detects them using two different methods
4. Compares which method works better, and why

## The Simulated Faults

| Fault Type | Description | Real-World Cause | Anomaly Type |
|---|---|---|---|
| Sudden Spike | Power jumps to ~75-80mW for a few readings | Short circuit, electrostatic discharge, voltage glitch | Point anomaly |
| Gradual Drift | Power slowly climbs from 50mW to 65mW over 100 readings | Overheating, component aging | Collective anomaly |
| Sudden Drop | Power falls to ~25mW | Dead or stuck circuit block | Point anomaly |

Out of 2000 total readings, 135 (6.75%) are labeled as real faults.

## Methods Compared

### 1. Z-score (statistical baseline)
Calculates how many standard deviations each reading is from the overall average. Flags anything beyond ±3 standard deviations as anomalous. Simple, fast, interpretable, but only considers each point's distance from a single global average.

### 2. Isolation Forest (ML model)
An unsupervised ML algorithm that randomly splits the data repeatedly. Points that isolate quickly (in few splits) are flagged as anomalies, points that need many splits to isolate are considered normal. Unlike Z-score, it considers the overall structure of the data, not just distance from one average.

## Results

| Metric | Z-score | Isolation Forest |
|---|---|---|
| Anomalies flagged | 50 | 135 |
| Real faults caught | 50 | 108 |
| False alarms | 0 | 27 |
| Missed faults | 85 | 27 |
| Recall | 37.04% | 80.00% |

## What I Learned

Z-score is extremely conservative: it never raised a false alarm, but missed nearly two-thirds of real faults. It caught the sudden spikes and drop easily (since they're extreme relative to the overall average) but missed most of the gradual drift, since a mid-drift reading isn't far enough from the global average to cross the ±3 threshold.

Isolation Forest caught over twice as many real faults, including most of the drift, because it looks at how isolated a point is relative to the data's actual structure, not just one global average. This came at a cost: 27 false alarms.

This is a real trade-off, not a simple "ML wins" story. If missing a real fault is dangerous (like a fault that could damage hardware), Isolation Forest's higher recall is worth the extra false alarms. If false alarms are costly (like triggering unnecessary manufacturing shutdowns), Z-score's precision might be preferred.

## A Note on Contamination

Isolation Forest needs a `contamination` parameter, an estimate of what fraction of the data is anomalous. I used the true rate (6.75%) since I generated the data myself and knew it exactly. In a real-world setting, this rate usually isn't known in advance and would need to be estimated from historical failure data, or a different anomaly-scoring approach would be used instead. This is a real limitation of Isolation Forest worth being aware of.

## Other Approaches (Not Implemented)

Other unsupervised anomaly detection methods exist, such as DBSCAN (density-based clustering) and Local Outlier Factor (comparing local density around each point). I chose Isolation Forest for this project because it doesn't require tuning a distance threshold and scales well to larger datasets. Comparing these methods would be a good next step.

## Tech Stack
- Python
- NumPy, Pandas (data generation and handling)
- Matplotlib (visualization)
- scikit-learn (Isolation Forest)

## Files
- `generate_data.py` — simulates the power trace with injected faults
- `detect_zscore.py` — Z-score anomaly detector
- `detect_iforest.py` — Isolation Forest anomaly detector
- `power_trace.csv` — generated dataset
- `zscore_plot.png`, `iforest_plot.png` — result visualizations
