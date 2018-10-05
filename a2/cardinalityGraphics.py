import json
import matplotlib.pyplot as plt
import numpy as np
data = {}
with open('results_all.json') as f:
  data = json.load(f)

yLabels = []

for isAddition in [False, True]:
  for i in range(26):
    if i > 0:
      if isAddition:
        yLabels.append(i)
      else:
        yLabels.append(-i)
      if i == 25 and not isAddition:
        yLabels.reverse()
        yLabels.append(0)
cardinalities = []
runtimes = []
rmse = []
mae = []
chartConfigs = []


plt.figure()
legendLabel = []
legendHandles = []
for cardinality in data:
  print("lengths are", len(yLabels), len(data[cardinality]), yLabels)
  handle, = plt.plot(yLabels, data[cardinality])
  legendHandles.append(handle)
  legendLabel.append(str(cardinality))

plt.axis([-25, 25, 0, 1])
plt.xlabel("Distance from Cardinality")
plt.ylabel("RAE")
plt.title("Card")
plt.legend(legendHandles, legendLabel)
plt.savefig("test.png")

# y_pos = np.arange(len(yLabels))
# for chart in chartConfigs: