import json
import matplotlib.pyplot as plt
import numpy as np
data = {}
with open('results_all.json') as f:
  data = json.load(f)

print("results are", data)

objects = []
runtimes = []
for algo in data:
  objects.append(algo["name"])
  runtimes.append(np.mean(algo["runtime"]))

y_pos = np.arange(len(objects))

plt.bar(y_pos, runtimes, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Used Algorithm')
plt.title('Runtimes in seconds')
 
plt.show()