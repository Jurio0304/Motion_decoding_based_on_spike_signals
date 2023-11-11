import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

feat_choose = ['only_pos', 'only_vel', 'only_acc', 'pos_vel',
               'pos_acc', 'vel_acc', 'all']
path = './results/feat_choose/'

cc_all = []
for i in feat_choose:
    cc = np.load(path + i + '/AllResults.npy')
    cc_all.append(np.mean(cc, axis=(0, 2)).flatten())

    print(f'Median CC ({i}): {np.median(cc)}')
    print(f'Mean CC ({i}): {np.mean(cc)}')

# transform to DataFrame
cc_new = cc_all[0]
methods = []
for i in range(len(cc_all)):
    for j in range(len(cc_all[i])):
        methods.append(feat_choose[i])
    if i > 0:
        cc_new = np.concatenate((cc_new, cc_all[i]))

data = pd.DataFrame({
    'Pearson Correlation': cc_new,
    'Features': methods,
})

# -------------------------- Plot ------------------------ #
rect = [0.1, 0.12, 0.88, 0.82]
colors = [sns.color_palette('Set2')[i] for i in range(len(feat_choose))]
__line_width__ = 3
y_ticks = [0.2 * x for x in range(0, 5)]

plt.figure(figsize=(12, 8))
ax = plt.axes(rect)
ax = sns.boxplot(data=data, x='Features', y='Pearson Correlation', fliersize=10,
                 palette=colors, order=feat_choose, ax=ax, width=0.5, linewidth=__line_width__)
ax.tick_params(which='major', direction='out', length=15, width=__line_width__, labelsize=22, bottom=True)

ax.grid(False)
ax.set_axisbelow(True)
ax.set_facecolor('white')
ax.set_yticks(y_ticks)
ax.set_ylim(0, 0.8)

# The ticks
ax.xaxis.set_tick_params(width=__line_width__, length=10)
ax.yaxis.set_tick_params(width=__line_width__, length=10)
ax.xaxis.label.set_fontsize(25)
ax.yaxis.label.set_fontsize(25)
c = [a.set_fontsize(20) for a in ax.get_yticklabels()]
# Despite
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(__line_width__)
ax.spines['left'].set_linewidth(__line_width__)

filename = os.path.join(path, 'cc_box.png')
plt.savefig(filename, dpi=300)
plt.show()

sys.exit(0)
