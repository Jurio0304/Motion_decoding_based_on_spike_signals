import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from decoders import get_R2
from main import Evaluation

result_path = './results/'
kin_file = 'indy_20170124_01_kin_test.npy'
kin_pred_file = 'indy_20170124_01_kin_pred.npy'
decoder = ['Linear', 'KF', 'DNN', 'LSTM']

n_motion = 6

kin = []
kin_pred = []

kin.append(np.load(result_path + decoder[0] + '/' + kin_file))
kin.append(np.load(result_path + decoder[1] + '/' + kin_file))
kin.append(np.load(result_path + decoder[2] + '/' + kin_file))
kin.append(np.load(result_path + decoder[3] + '/' + kin_file))

kin_pred.append(np.load(result_path + 'Linear/' + kin_pred_file))
kin_pred.append(np.load(result_path + 'KF/' + kin_pred_file))
kin_pred.append(np.load(result_path + 'DNN/' + kin_pred_file))
kin_pred.append(np.load(result_path + 'LSTM/' + kin_pred_file))

all_cc = np.zeros((len(kin), n_motion))
all_r2 = np.zeros(all_cc.shape)

for i in range(len(kin)):
    r2 = get_R2(kin[i], kin_pred[i])
    for MoBin in range(n_motion):
        all_r2[i, MoBin] = r2[MoBin]

        r, p = pearsonr(kin[i][:, MoBin], kin_pred[i][:, MoBin])
        all_cc[i, MoBin] = r

# # Plot all cc
# Evaluation(result_path, all_cc, decoder)
# np.save(os.path.join(result_path, 'all_cc.npy'), all_cc)

# Plot all r2
a = 0.7
colors = ['C' + str(i) for i in range(len(kin))]
x = range(len(kin))

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.bar(x, all_r2[:, 0], alpha=a, color=colors)
plt.xlabel('x_pos')
plt.ylabel('R^2')

plt.subplot(2, 3, 2)
plt.bar(x, all_r2[:, 2], alpha=a, color=colors)
# for i in range(len(kin)):
#     plt.bar(x[i], all_r2[i, 2], alpha=a, color=colors[i], label=decoder[i])
plt.xlabel('x_vel')

plt.subplot(2, 3, 3)
# plt.bar(x, all_r2[:, 4], alpha=a, color=colors)
for i in range(len(kin)):
    plt.bar(x[i], all_r2[i, 4], alpha=a, color=colors[i], label=decoder[i])
plt.xlabel('x_acc')
plt.legend()

plt.subplot(2, 3, 4)
plt.bar(x, all_r2[:, 1], alpha=a, color=colors)
plt.xlabel('y_pos')

plt.subplot(2, 3, 5)
plt.bar(x, all_r2[:, 3], alpha=a, color=colors)
plt.xlabel('y_vel')

plt.subplot(2, 3, 6)
plt.bar(x, all_r2[:, 5], alpha=a, color=colors)
plt.xlabel('y_acc')

plt.savefig(result_path + '/all_r2.png', dpi=600)
plt.show()
np.save(os.path.join(result_path, 'all_r2.npy'), all_r2)
