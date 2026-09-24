import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
df=pd.read_csv("Power_trace.csv")
x=df[["power_mw"]].values.reshape(-1,1)
model=IsolationForest(contamination=0.0675,random_state=0)
model.fit(x)
df["iforest_pred"]=model.predict(x)
anomalies=df[df["iforest_pred"]==-1]
print("Total readings:", len(df))
print("Anomalies detected:", len(anomalies))

true_faults = df["label"].sum()
caught_faults = anomalies["label"].sum()
false_alarms = len(anomalies) - caught_faults
missed_faults = true_faults - caught_faults

print("True faults in data:", true_faults)
print("Faults caught:", caught_faults)
print("False alarms:", false_alarms)
print("Missed faults:", missed_faults)
print(f"Recall: {caught_faults/true_faults:.2%}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df["time"], df["power_mw"], color="blue", label="Power Trace")
plt.scatter(anomalies["time"], anomalies["power_mw"], color="orange", label="Isolation Forest Anomalies")
plt.xlabel("Time")
plt.ylabel("Power (mW)")
plt.title("Isolation Forest Anomaly Detection")
plt.legend()
plt.savefig("iforest_plot.png")
plt.show()