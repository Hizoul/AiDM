import json
import matplotlib.pyplot as plt
import numpy as np
data = {}
with open('results_all.json') as f:
  data = json.load(f)

print("results are", data)

objects = []
runtimes = []
rmse = []
mae = []
for algo in data:
  objects.append(algo["name"])
  runtimes.append(np.mean(algo["runtime"]))
  rmse.append(np.mean(algo["rmses"]))
  mae.append(np.mean(algo["maes"]))

chartConfigs = [
  {"yLabel": "Seconds", "fileName": "runtime", "name": "Runtimes in Seconds", "data": runtimes},
  {"yLabel": "RMSE", "fileName": "rmse", "name": "Root Mean Squared Error", "data": rmse},
  {"yLabel": "MAE", "fileName": "mae", "name": "Mean Absolute Error", "data": mae}
]

y_pos = np.arange(len(objects))
for chart in chartConfigs:
  plt.figure()
  plt.bar(y_pos, chart["data"], align='center', alpha=0.5)
  plt.xticks(y_pos, objects)
  plt.xlabel("Algorithm Name")
  plt.ylabel(chart["yLabel"])
  plt.title(chart["name"])
  plt.savefig(chart["fileName"])
  print("Data is ", chart["data"])