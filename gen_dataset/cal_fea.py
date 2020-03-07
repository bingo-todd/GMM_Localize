import numpy as np
import os
import sys
from multiprocessing import Process, Manager

my_modules_dir = os.path.expanduser('~/my_modules')
sys.path.append(os.path.join(my_modules_dir, 'Auditory_model'))
from Auditory_model import Auditory_model  # noqa: E402

sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import wav_tools  # noqa: E402
import plot_tools  # noqa: E402
from get_fpath import get_fpath  # noqa: E402
from ProcessBarMulti import *  # noqa: E402


fs = 44100
freq_low = 80
freq_high = 5000

frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)
n_band = 32


def plot_cue_sample(cues, ax):
    # fig,ax = plt.subplots(1,1)
    ax.scatter(cues[:, 0], cues[:, 1], alpha=0.3)
    ax.set_xlabel('ITD(ms)')
    ax.set_ylabel('ILD(dB)')
    # return fig


def cal_itd_ild_sub(record_tar_fpath, record_inter_fpath, fea_fpath,
                    front_end):

    if os.path.exists(fea_fpath):
        return

    record_tar, fs = wav_tools.wav_read(record_tar_fpath)
    record_inter, fs = wav_tools.wav_read(record_inter_fpath)
    min_len = min((record_tar.shape[0], record_inter.shape[0]))

    [cue_frame_all,
     ccf_frame_all,
     snr_frame_all] = front_end.cal_cues(
                    tar=record_tar[:min_len], inter=record_inter[:min_len],
                    frame_len=frame_len, shift_len=shift_len,
                    max_delay=max_delay, n_worker=1)

    np.savez(fea_fpath,
             cue_frame_all=cue_frame_all,
             ccf_frame_all=ccf_frame_all,
             snr_frame_all=snr_frame_all)


def parallel_worker(func, arg_all, pb_share):
    front_end = Auditory_model(fs=fs, freq_low=freq_low, freq_high=freq_high,
                               n_band=32, is_middle_ear=True,
                               ihc_type='Roman')
    while not arg_all.empty():
        arg = arg_all.get_nowait()
        func(*arg, front_end=front_end)
        pb_share.update()


def cal_itd_ild(record_dir_base, fea_dir_base, room_all, mic_pos_all):
    azi_tar_all = range(8, 29)
    azi_diff_all = [-8, -6, -4, -2, -1, 1, 2, 4, 6, 8]
    snr_all = [0, 10, 20]

    n = (len(room_all) * len(mic_pos_all) * len(azi_tar_all)
         * len(azi_diff_all) * len(snr_all))
    pb_share = ProcessBarMulti([n], desc_all=['cal_cues'])

    arg_all = Manager().Queue()
    for room in room_all:
        for mic_pos in mic_pos_all:
            record_dir = os.path.join(record_dir_base, room, str(mic_pos))
            fea_dir = os.path.join(fea_dir_base, room, str(mic_pos))
            if not os.path.exists(fea_dir):
                os.makedirs(fea_dir)
            for azi_tar in azi_tar_all:
                for azi_diff in azi_diff_all:
                    azi_inter = azi_tar+azi_diff
                    for snr in snr_all:
                        condition_str = '{}_{}_{}'.format(azi_tar, azi_inter,
                                                          snr)

                        record_tar_fpath = os.path.join(
                                        record_dir,
                                        '{}_tar.wav'.format(condition_str))

                        record_inter_fpath = os.path.join(
                                        record_dir,
                                        '{}_inter.wav'.format(condition_str))

                        fea_fpath = os.path.join(
                                    fea_dir, '{}.npz'.format(condition_str))

                        arg_all.put((record_tar_fpath,
                                     record_inter_fpath,
                                     fea_fpath))

    n_worker = 24
    proc_all = []
    for i in range(n_worker):
        proc = Process(target=parallel_worker,
                       args=(cal_itd_ild_sub, arg_all, pb_share),
                       daemon=True)
        proc.start()
        proc_all.append(proc)

    [proc.join() for proc in proc_all]


if __name__ == '__main__':

    cal_itd_ild(record_dir_base='Data/Records/train',
                fea_dir_base='Data/Features/train',
                room_all=['RT_0.5'],
                mic_pos_all=[1, 3, 5, 7, 9, 11, 13, 15])
