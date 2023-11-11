import logging
import os
import sys

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')

from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from decoders import DNNDecoder, KFDecoder, LSTMDecoder
from preprocessing_funcs import get_spikes_with_history


def train_decoder(x_train, y_train, x_test, y_test, path, de='Linear', units=100, epochs=10):
    """
    Train different decoder: Linear, DNN, LSTM
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param rec:
    :param de:
    :return:
    """
    global y_pred, decoder
    verbose = 1

    if de == 'Linear':
        decoder = LinearRegression(n_jobs=5)
        # Fit the regression model
        decoder.fit(x_train, y_train)
        # Predict the reconstructed spectrogram for the test data
        y_pred = decoder.predict(x_test)

    elif de == 'KF':
        decoder = KFDecoder(C=1)
        decoder.fit(x_train, y_train)
        y_pred = decoder.predict(x_test, y_test)

    elif de == 'DNN':
        decoder = DNNDecoder(units=units, dropout=0, num_epochs=epochs, verbose=verbose)
        decoder.fit(x_train, y_train, valid_data=(x_test, y_test), logger=logger)
        y_pred = decoder.predict(x_test)

    elif de == 'LSTM':
        decoder = LSTMDecoder(units=units, dropout=0, num_epochs=epochs, verbose=verbose)
        decoder.fit(x_train[:, np.newaxis, :], y_train, valid_data=(x_test[:, np.newaxis, :], y_test),
                    logger=logger)
        y_pred = decoder.predict(x_test[:, np.newaxis, :])

    if de != 'Linear' and de != 'KF':
        index = np.arange(epochs)
        dataframe = pd.DataFrame({
            'epochs': index,
            'cc_train': decoder.cc_train,
            'cc_test': decoder.cc_valid,
            'loss_train': decoder.loss_train,
            'loss_test': decoder.loss_valid,
            'r2_x_pos': decoder.r2_x_pos,
            'r2_y_pos': decoder.r2_y_pos,
            'r2_x_vel': decoder.r2_x_vel,
            'r2_y_vel': decoder.r2_y_vel,
            'r2_x_acc': decoder.r2_x_acc,
            'r2_y_acc': decoder.r2_y_acc,
        })
        dataframe.to_csv(path + '/train_log.csv', index=False, sep=',')

    return y_pred


def Evaluation(result_path, allRes, pts):
    colors = ['C' + str(i) for i in range(10)]

    meanCorrs = np.mean(allRes, axis=(1, 2))
    stdCorrs = np.std(allRes, axis=(1, 2))

    x = range(len(meanCorrs))

    # 创建一个宽12，高6的空白图像区域
    fig = plt.figure(figsize=(12, 6))
    # rect可以设置子图的位置与大小
    rect1 = [0.10, 0.15, 0.25, 0.80]  # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect2 = [0.45, 0.15, 0.50, 0.80]
    # 在fig中添加子图ax，并赋值位置rect
    ax = plt.axes(rect1), plt.axes(rect2)

    # Barplot of average results
    ax[0].bar(x, meanCorrs, yerr=stdCorrs, alpha=0.5, color=colors)
    for p in range(allRes.shape[0]):
        # Add mean results of each patient as scatter points
        ax[0].scatter(np.zeros(allRes[p, :, :].shape[0]) + p, np.mean(allRes[p, :, :], axis=1), color=colors[p])

    ax[0].set_xticks(x)
    ax[0].set_xticklabels([pts[i] for i in x], rotation=30, ha='right', fontsize=20)
    ax[0].set_ylim(0.3, 1)
    ax[0].set_ylabel('Pearson Correlation')
    # Title
    ax[0].set_title('a', fontsize=20, fontweight="bold")
    # Make pretty
    plt.setp(ax[0].spines.values(), linewidth=2)
    # The ticks
    ax[0].xaxis.set_tick_params(width=2)
    ax[0].yaxis.set_tick_params(width=2)
    ax[0].xaxis.label.set_fontsize(20)
    ax[0].yaxis.label.set_fontsize(20)
    c = [a.set_fontsize(20) for a in ax[0].get_yticklabels()]

    # Despine
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    # Mean across folds over spectral bins
    specMean = np.mean(allRes, axis=1)
    specStd = np.std(allRes, axis=1)
    specBins = np.arange(allRes.shape[2])
    for p in range(allRes.shape[0]):
        ax[1].plot(specBins, specMean[p, :], color=colors[p], linewidth=1)
        error = specStd[p, :] / np.sqrt(allRes.shape[1])
        # Shaded areas highlight standard error
        ax[1].fill_between(specBins, specMean[p, :] - error, specMean[p, :] + error, alpha=0.5, color=colors[p])
    ax[1].set_ylim(0.3, 1)
    ax[1].set_xlim(0, len(specBins) - 1)
    ax[1].set_xlabel('Motion')
    ax[1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[1].set_xticklabels(['x_pos', 'y_pos', 'x_vel', 'y_vel', 'x_acc', 'y_acc'], rotation=45, ha='right', fontsize=20)
    ax[1].set_ylabel('Pearson Correlation')
    # Title
    ax[1].set_title('b', fontsize=20, fontweight="bold")

    # Make pretty
    plt.setp(ax[1].spines.values(), linewidth=2)
    # The ticks
    ax[1].xaxis.set_tick_params(width=2)
    ax[1].yaxis.set_tick_params(width=2)
    ax[1].xaxis.label.set_fontsize(20)
    ax[1].yaxis.label.set_fontsize(20)
    c = [a.set_fontsize(20) for a in ax[1].get_yticklabels()]
    c = [a.set_fontsize(20) for a in ax[1].get_xticklabels()]
    # Despine
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'results.png'), dpi=600)
    plt.show()


def Plot_data(neural_data, vels_binned, path):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(neural_data.T, aspect='auto', origin='lower')
    plt.xlim([0, neural_data.shape[0]])
    plt.ylabel('neuron_id')
    plt.title('neural activity')
    plt.subplot(2, 1, 2)
    plt.plot(vels_binned)
    plt.xlabel('time')
    plt.xlim([0, vels_binned.shape[0]])
    plt.title('velocity')
    plt.savefig(path + '/fr_motion.png', dpi=600)


def Plot_pred(y, y_pre, path, k=1):
    show_len = 100
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(y[:show_len, 0])
    plt.plot(y_pre[:show_len, 0])
    plt.xlabel('x_pos')
    plt.subplot(2, 3, 2)
    plt.plot(y[:show_len, 2])
    plt.plot(y_pre[:show_len, 2])
    plt.xlabel('x_vel')
    plt.subplot(2, 3, 3)
    plt.plot(y[:show_len, 4])
    plt.plot(y_pre[:show_len, 4])
    plt.xlabel('x_acc')
    plt.subplot(2, 3, 4)
    plt.plot(y[:show_len, 1])
    plt.plot(y_pre[:show_len, 1])
    plt.xlabel('y_pos')
    plt.subplot(2, 3, 5)
    plt.plot(y[:show_len, 3])
    plt.plot(y_pre[:show_len, 3])
    plt.xlabel('y_vel')
    plt.subplot(2, 3, 6)
    plt.plot(y[:show_len, 5])
    plt.plot(y_pre[:show_len, 5])
    plt.xlabel('y_acc')
    plt.savefig(path + f'/motion_pred_{k + 1}.png', dpi=600)


def preprocess(fr, motion):
    bins_before = 6  # How many bins of neural data prior to the output are used for decoding
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = 6  # How many bins of neural data after the output are used for decoding

    # Function to get the covariate matrix that includes spike history from previous bins
    x = get_spikes_with_history(fr, bins_before, bins_after, bins_current)
    y = motion
    # Put in "flat" format, so each "neuron / time" is a single feature
    x_flat = x.reshape(x.shape[0], (x.shape[1] * x.shape[2]))

    # Set what part of data should be part of the training/testing/validation sets
    training_range = [0, 1]
    # testing_range = [0.8, 1]
    # valid_range = [0.8, 0.9]
    num_examples = x.shape[0]

    training_set = np.arange(np.int32(np.round(training_range[0] * num_examples)) + bins_before,
                             np.int32(np.round(training_range[1] * num_examples)) - bins_after)

    # Get training data
    x_train = x[training_set, :, :]
    x_flat_train = x_flat[training_set, :]
    y_train = y[training_set, :]

    # Z-score "X" inputs.
    x_train_mean = np.nanmean(x_train, axis=0)
    x_train_std = np.nanstd(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std

    # Z-score "X_flat" inputs.
    x_flat_train_mean = np.nanmean(x_flat_train, axis=0)
    x_flat_train_std = np.nanstd(x_flat_train, axis=0)
    x_flat_train = (x_flat_train - x_flat_train_mean) / x_flat_train_std

    # Zero-center outputs
    y_train_mean = np.mean(y_train, axis=0)
    y_train = y_train - y_train_mean

    X_train_std = np.nanstd(x_train, axis=0)
    X_flat_train_std = np.nanstd(x_flat_train, axis=0)
    y_train = y_train - y_train_mean

    return x_flat_train, y_train


def Decoding(logger=None):
    logger.info('Decoding...')
    # decoder = ['Linear', 'KF', 'DNN', 'LSTM']
    decoder = ['KF']

    # ---------------------------------- #
    # de = 3  # 0:Linear, 1:KF, 2:DNN, 3:LSTM
    units = 200  # 解码器隐藏层大小
    epochs = 20
    # ---------------------------------- #

    path = r'./results/'
    feat_path = r'./data/'
    pts = ['indy_20170124_01']

    n_motion = 6
    nfolds = 10
    kf = KFold(nfolds, shuffle=False)

    # Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    allRes = np.zeros((len(decoder), nfolds, n_motion))

    for de in range(len(decoder)):
        # decoder: 0:Linear, 1:KF, 2:DNN, 3:LSTM
        result_path = path + decoder[de]
        os.makedirs(os.path.join(result_path), exist_ok=True)

        for pNr, pt in enumerate(pts):
            # Load the data
            f = h5py.File(feat_path + pt + '_fr.mat')
            threshold = 1  # 阈值设为1Hz
            fr_feat = np.array(f['fr'])
            id_good = np.where(fr_feat.mean(axis=1) > threshold)[0]  # 排除实验中平均发放率小于1Hz的神经元
            fr_feat = fr_feat[id_good].T

            f = h5py.File(feat_path + pt + '_kin.mat')
            motion_feat = f['kin'][:].T

            # # Plot original data
            # Plot_data(fr_feat, motion_feat, result_path)

            logger.info(f'Decoder [{decoder[de]}] Preprocessing...')

            # # Choose features
            # fr_feat, motion_feat = fr_feat[:, 2:4], motion_feat[:, 2:4]
            # fr_feat, motion_feat = (np.hstack((fr_feat[:, :2], fr_feat[:, 4:])),
            #                         np.hstack((motion_feat[:, :2], motion_feat[:, 4:])))

            # Save the correlation coefficients for each fold
            rs = np.zeros((nfolds, motion_feat.shape[1]))
            rec_motion = np.zeros(motion_feat.shape)

            if decoder[de] == 'KF':
                # if KF, no preprocess
                data_x, data_y = fr_feat, motion_feat
            else:
                data_x, data_y = preprocess(fr_feat, motion_feat)

            y_test, y_test_pred = None, None
            for k, (train, test) in enumerate(kf.split(data_x)):
                logger.info(f'---------- {decoder[de]}_k-{k + 1} processing: ----------')

                x_train, y_train, x_test, y_test = \
                    data_x[train, :], data_y[train, :], data_x[test, :], data_y[test, :]

                y_test_pred = train_decoder(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                            path=result_path, de=decoder[de], units=units, epochs=epochs)

                Plot_pred(y_test, y_test_pred, result_path, k)

                # Evaluate reconstruction of this fold
                motion_feat[test, :] = y_test
                rec_motion[test, :] = y_test_pred
                for MoBin in range(motion_feat.shape[1]):
                    if np.any(np.isnan(y_test_pred)):
                        logger.info('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(y_test_pred))))
                    r, p = pearsonr(y_test[:, MoBin], y_test_pred[:, MoBin])
                    rs[k, MoBin] = r

            # Show evaluation result
            allRes[de, :, :] = rs
            logger.info(f'%s has mean correlation of %f ({decoder[de]})' % (pt, np.mean(rs)))

            np.save(os.path.join(result_path, f'{pt}_kin_test.npy'), motion_feat)
            np.save(os.path.join(result_path, f'{pt}_kin_pred.npy'), rec_motion)

        np.save(os.path.join(path, 'AllResults.npy'), allRes)

    logger.info('\n\nEvaluate...')
    Evaluation(result_path=path, allRes=allRes, pts=decoder)
    logger.info('Done!\n\n')

    return allRes


if __name__ == "__main__":
    # -----------test----------- #
    # 记录日志
    logger = logging.getLogger()
    log_file = 'Decoding.log'
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, 'w+'),
            logging.StreamHandler(sys.stdout)
        ])

    allRes = Decoding(logger=logger)
