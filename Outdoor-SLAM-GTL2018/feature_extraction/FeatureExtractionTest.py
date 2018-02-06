import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAX_DIST = 40
DIFF_THRES = 15
MAX_WITDH = 30  # needs tuning


df = pd.read_csv("test.csv")

# select a random line
i = 12000
ranges = df.select(lambda col: col.startswith('field.ranges'), axis=1).values[i, :]
START_ANGLE = df['field.angle_min'][i]
ANGLE_INCREMENT = df['field.angle_increment'][i]

# clean the data
ranges[ranges == 0] = MAX_DIST
ranges[ranges > MAX_DIST] = MAX_DIST

# compute the differences and locate features
diffs = []
features_map = []
look_for_neg_peak = True
current_feature = {}
for i in range(len(ranges)-1):
    diffs.append(ranges[i+1]-ranges[i])
    if look_for_neg_peak:
        if diffs[i] < -DIFF_THRES:
            current_feature['neg_i'] = i
            look_for_neg_peak = False
    else:
        if diffs[i] > DIFF_THRES:
            current_feature['pos_i'] = i
            features_map.append(current_feature)
            current_feature = {}
            look_for_neg_peak = True
        elif i - current_feature['neg_i'] > MAX_WITDH:
            current_feature = {}
            look_for_neg_peak = True

print(features_map)
# extract more info about features
for feature in features_map:
    theta_index = 0.5*(feature['pos_i'] + feature['neg_i'])
    feature['theta'] = theta_index*ANGLE_INCREMENT
    feature['d'] = ranges[int(theta_index)] + np.math.sin(ANGLE_INCREMENT*0.5*(feature['pos_i'] - feature['neg_i']))

print(features_map)

plt.plot(ranges)
plt.ylabel("ranges")
plt.show()
plt.plot(diffs)
plt.ylabel("differences")
plt.show()