import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Lock, Queue  # noqa
import warnings
import os
import sys
from BasicTools import wav_tools, plot_tools, get_fpath, Filter_GPU
from GMMs import GMMs
from Auditory_model import Auditory_model
from GMM_Localizer import GMM_Localizer

# constant variables

fs = 44100
n_band = 32
freq_low = 80
freq_high = 5000
frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)
azi_sep_theta = 2


room_all = ['RT_0.19', 'RT_0.29', 'RT_0.39', 'RT_0.48',
            'RT_0.58', 'RT_0.69']
mic_label_test_all = [2, 4, 5, 6, 8, 10, 12, 14]
azi_tar_all = np.arange(8, 29)
n_inter_all = [0, 1, 2, 3]

n_test = 3
n_room = len(room_all)
n_mic_pos = len(mic_label_test_all)
n_azi_tar = azi_tar_all.shape[0]
n_n_inter = len(n_inter_all)

front_end = Auditory_model(fs, freq_low=freq_low, freq_high=freq_high,
                           n_band=n_band, is_middle_ear=True,
                           ihc_type='Roman')


def load_test_sample(set_dir, room, mic_label, azi_tar, n_inter, test_i):

    mix_fpath = os.path.join(set_dir,
                             '_'.join((f'{room}', f'{mic_label}',
                                       f'{azi_tar}', f'{n_inter}',
                                       f'{test_i}.npy')))
    mix = np.load(mix_fpath)

    cues, ccfs = front_end.cal_cues(tar=mix, frame_len=frame_len,
                                    shift_len=shift_len, max_delay=max_delay,
                                    n_worker=1)
    return cues


def evaluate_mic_sub(model, filter_gpu, set_dir, room, mic_label, azi_tar,
                     result_dir):

    # result_fpath = os.path.join(result_dir, f'{room}_{mic_label}.npz')
    # if os.path.exists(result_fpath):
    #     warnings.warn(f'{result_fpath}: has existed')

    for n_inter_i, n_inter in enumerate(n_inter_all):
        for test_i in range(n_test):
            raw_result_fpath = os.path.join(
                            result_dir,
                            '{}_{}_{}_{}_{}.npz'.format(room, mic_label,
                                                        azi_tar, n_inter,
                                                        test_i))
            if os.path.exists(raw_result_fpath):
                warnings.warn(f'{raw_result_fpath}: has existed')
                # os.remove(raw_result_fpath)
                # continue

            [cues, azi_gt_all] = load_test_sample(filter_gpu, set_dir, room,
                                                 mic_label, azi_tar, n_inter,
                                                 test_i)
            azi_est = model.locate(cues)
            print(azi_est, azi_gt_all)
            raise Exception()
            np.savez(raw_result_fpath, azi_est=azi_est,
                     azi_gt_all=azi_gt_all)

    #             azi_diff = np.min(np.abs(np.subtract(
    #                                         azi_gt_all.reshape([-1,1]),
    #                                         azi_est)),
    #                               axis=0)
    #
    #             n_frame = azi_est.shape[0]
    #             n_frame_correct = np.count_nonzero(azi_diff<=azi_theta)
    #             cp_tmp = np.float32(n_frame_correct)/n_frame
    #             cp_all[azi_tar_i,n_inter_i,test_i] = cp_tmp
    #
    #             rmse_all[azi_tar_i,
    #                      n_inter_i,test_i] = np.sqrt(np.sum(azi_diff**2))
    #
    # np.savez(result_fpath,cp_all=cp_all,rmse_all=rmse_all)


def parallel_worker(arg_all, model_dir, pb_share):
    filter_gpu = Filter_GPU.Filter_GPU(0)
    model = GMM_Localizer(model_dir=model_dir)
    while True:
        try:
            arg = arg_all.get_nowait()
            if arg is not None:
                evaluate_mic_sub(model, filter_gpu, *arg)
                pb_share.update()
            else:
                return
        except Exception:
            return


def evaluate_main(n_worker):
    test_set_dir = 'Data/Records/test'
    model_dir = 'models/all_room'

    filter_gpu = Filter_GPU.Filter_GPU(0)
    model = GMM_Localizer(model_dir=model_dir)
    for room in room_all:
        for mic_label in mic_label_test_all:
            for azi_tar in azi_tar_all:
                evaluate_mic_sub(model, filter_gpu, test_set_dir, room,
                                 mic_label, azi_tar, 'Result_raw/all_room')


def plot_result():
    rmse_all = np.zeros((n_room, n_mic_pos, n_azi_tar, n_n_inter, n_test))
    cp_all = np.zeros((n_room, n_mic_pos, n_azi_tar, n_n_inter, n_test))

    for room_i, room in enumerate(room_all):
        for mic_pos_i, mic_label in enumerate(mic_label_test_all):
            result_fpath = 'Result/all_room/{}_{}.npz'.format(room, mic_label)
            result_tmp = np.load(result_fpath)
            rmse_all[room_i, mic_pos_i] = result_tmp['rmse_all']
            cp_all[room_i, mic_pos_i] = result_tmp['cp_all']

    #
    fig, ax = plt.subplots(1, 2, sharex=True)
    for n_inter_i in range(n_n_inter):
        ax[0].plot(range(n_room), np.mean(rmse_all[:, :, :, n_inter_i, :],
                                          axis=(1, 2, 3)))
        ax[1].plot(range(n_room), np.mean(cp_all[:, :, :, n_inter_i, :],
                                          axis=(1, 2, 3)))
    ax[0].set_xticklabels(room_all)
    plot_tools.savefig(fig, name='result_all.png', dir='./')


if __name__ == '__main__':

    # filter_gpu = Filter_GPU(0)
    # room = 'RT_0.19'
    # azi_tar=18
    # n_inter = 1
    # mic_label = 5
    # for i in range(5):
    #     gen_test_sample(room,mic_label,azi_tar,n_inter,filter_gpu)

    # model = GMM_Localizer(model_dir='models/all_room')
    # for n_iter in range(0,4):
    #     test('RT_0.19',5,8,n_iter,model)
    #     test('RT_0.19',5,18,n_iter,model)
    #     test('RT_0.19',5,28,n_iter,model)

    # n_worker = int(sys.argv[1])
    # evaluate_main(n_worker)

    if True:
        print('load model')
        model = GMM_Localizer(model_dir='models/all_room')

        set_dir = 'Data/Records/test'
        room = 'RT_0.69'
        mic_label = 5
        n_inter = 0
        azi_theta = 1

        cp_all = np.zeros(n_azi_tar)
        rmse_all = np.zeros(n_azi_tar)

        fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True)
        for i in range(n_azi_tar):
            azi_tar = azi_tar_all[i]
            print('synthesizing')
            [record_tar,
             cue_all,
             src_azi_all] = load_test_sample(set_dir, room, mic_label, azi_tar,
                                             n_inter, 0)
            print('prefict')
            azi_est = model.locate(cue_all)
            ax[0].plot(record_tar)
            n_sample = azi_est.shape[0]
            ax[1].plot(np.arange(n_sample)*shift_len+frame_len/2, azi_est)
            print(src_azi_all)
            # plt.show(block=False)
            plt.savefig(f'{azi_tar}.png')
            # input()
            ax[0].clear()
            ax[1].clear()
