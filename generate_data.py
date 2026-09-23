import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)
n=2000
time=np.arange(2000)
power_mw=np.random.normal(50,1.5,n)
label = np.zeros(n, dtype=int)
for i in [300,750,1400]:
    power_mw[i:i+5]+=np.random.uniform(20,30)
    label[i:i+5]=1
power_mw[1000:1100] += np.linspace(0, 15, 100)
label[1000:1100] = 1
power_mw[1700:1720] -= 25
label[1700:1720] = 1
df=pd.DataFrame({"time":time,"power_mw":power_mw,"label":label})
df.to_csv("Power_trace.csv",index=False)
print("Saved Power_trace.csv")
#print(df.describe())



df = pd.read_csv("power_trace.csv")

plt.plot(df["time"], df["power_mw"], label="Power", linewidth=0.8)

anomalies = df[df["label"] == 1]
plt.scatter(anomalies["time"], anomalies["power_mw"], color="red", s=10,
             label="Injected fault")
plt.xlabel("Time (samples)")
plt.ylabel("Power (mW)")
plt.title("Simulated Chip Power Trace")
plt.legend()
plt.show()
