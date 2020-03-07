import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Lock, Queue  # noqa
import warnings
import os
import sys
from BasicTools import wav_tools, plot_tools, get_fpath, Filter_GPU
from GMMs import GMMs


class GMM_Localizer:
    """"""
    def __init__(self, model_dir):

        azi_all = np.arange(8, 29)  # [-50,50]
        n_azi = azi_all.shape[0]
        n_band = 32
        model_all = np.ndarray((n_azi, n_band), dtype=np.object)
        for azi_i, azi in enumerate(azi_all):
            for band_i in range(n_band):
                model_fpath = os.path.join(
                                model_dir, '{}_{}.npy'.format(azi, band_i))
                model_all[azi_i, band_i] = GMMs()
                model_all[azi_i, band_i].load(model_fpath)

        self.n_band = n_band
        self.azi_all = azi_all
        self.azi_resolution = 5
        self.model_all = model_all

    def cal_lh(self, x):
        """
        x: [n_band,n_sample,fea_len]
        """
        epsilon = 1e-20

        n_azi = self.azi_all.shape[0]
        n_band = self.n_band

        # maximum likelihood
        n_sample = x.shape[1]
        prob = np.zeros((n_sample, n_azi, n_band))
        for azi_i in range(n_azi):
            for band_i in range(n_band):
                model_tmp = self.model_all[azi_i, band_i]
                prob[:, azi_i, band_i] = model_tmp.cal_prob(x[band_i])

        # prob of bands are multipled together
        lh = np.sum(np.log(prob+epsilon), axis=2)
        return lh

    def locate(self, x):
        """
        x: [n_band,n_sample,fea_len]
        """
        epsilon = 1e-20

        n_azi = self.azi_all.shape[0]
        n_band = self.n_band

        # maximum likelihood
        n_sample = x.shape[1]
        prob = np.zeros((n_sample, n_azi, n_band))
        for azi_i in range(n_azi):
            for band_i in range(n_band):
                model_tmp = self.model_all[azi_i, band_i]
                prob[:, azi_i, band_i] = model_tmp.cal_prob(x[band_i])

        # prob of bands are multipled together
        lh = np.sum(np.log(prob+epsilon), axis=2)
        max_pos = self._get_max_pos(lh)
        azi_est = (max_pos + 8)
        return azi_est

    def _get_max_pos(self, lh):
        n_frame, lh_len = lh.shape
        max_pos_all = np.zeros(n_frame)
        for frame_i in range(n_frame):
            lh_frame = lh[frame_i]
            lh_frame = lh_frame-np.min(lh_frame)+1.0

            max_pos_tmp = np.argmax(lh_frame)
            if max_pos_tmp >= 1 and max_pos_tmp <= lh_frame.shape[0]-2:
                [value_l, value_m,
                 value_r] = np.log(lh_frame[max_pos_tmp-1:max_pos_tmp+2])
                delta = (value_r-value_l)/(2*(2*value_m-value_l-value_r))
            else:
                delta = 0
            max_pos_all[frame_i] = max_pos_tmp+delta

        return max_pos_all


