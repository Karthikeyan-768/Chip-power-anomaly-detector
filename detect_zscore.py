import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Power_trace.csv")
mean=df["power_mw"].mean()
std=df["power_mw"].std()
df["zscore"]=(df["power_mw"]-mean)/std
anomalies=df[abs(df["zscore"])>3]
print("Total Readings:",len(df))
print("Anomalies Detected:",len(anomalies))
print(anomalies[["time","power_mw","zscore","label"]])
print(anomalies[anomalies["time"] >= 1700])
true_faults = df["label"].sum()
caught_faults = anomalies["label"].sum()
false_alarms = len(anomalies) - caught_faults
missed_faults = true_faults - caught_faults

print(f"True faults in data: {true_faults}")
print(f"Faults caught: {caught_faults}")
print(f"False alarms: {false_alarms}")
print(f"Missed faults: {missed_faults}")
print(f"Recall: {caught_faults/true_faults:.2%}")

plt.figure(figsize=(12, 5))
plt.plot(df["time"], df["power_mw"], color="blue", label="Power Trace")
plt.scatter(anomalies["time"], anomalies["power_mw"], color="red", label="Z-score Anomalies")
plt.xlabel("Time")
plt.ylabel("Power (mW)")
plt.title("Z-score Anomaly Detection")
plt.legend()
plt.savefig("zscore_plot.png")
plt.show()