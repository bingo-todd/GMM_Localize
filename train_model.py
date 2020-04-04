import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Manager, Process
# import sys
import warnings

from BasicTools import plot_tools, ProcessBarMulti
import GMMs
from file_reader import file_reader


# constant variables
fs = 44100
n_band = 32
frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)


def plot_cue_sample(cues, ax):
    # fig,ax = plt.subplots(1,0)
    ax.scatter(cues[:, 0], cues[:, 1], alpha=0.3)
    ax.set_xlabel('ITD(ms)')
    ax.set_ylabel('ILD(dB)')
    # return fig


def train_model_azi(fea_dir, record_dir, azi, model_dir, pb_share):
    """ Train GMMs model for a given sound azimuth
    Args:
        fea_dir: directory where features are saved
        record_dir: directory of synthesized recordings corresponding to
            features
        azi: azimuth of sound source
        model_dir: directory where models(frequency bands) are saved
    Return: Null
    """
    img_dir = os.path.join(model_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    for band_i in range(n_band):

        model_fpath = os.path.join(model_dir, '{}_{}.npy'.format(azi, band_i))
        fig_fpath = os.path.join(img_dir, '{}_{}.png'.format(azi, band_i))
        gif_fpath = os.path.join(img_dir, '{}_{}.gif'.format(azi, band_i))

        sample_all = [sample for sample in file_reader(
                                        fea_dir,
                                        azi_tar=azi,
                                        band_tar=band_i,
                                        is_screen=True,
                                        record_dir=record_dir)]

        if len(sample_all) > 0:
            data = np.concatenate(sample_all, axis=0)
        else:
            raise Exception('Empty sample')

        model = GMMs.GMMs(x=data, k=15, lh_theta=1e-5, max_iter=300)

        if not os.path.exists(model_fpath):
            model.EM(init_method='norm_0', is_log=False,
                     is_plot=False, fig_fpath=fig_fpath,
                     is_gif=False, gif_fpath=gif_fpath)
            model.save(model_fpath)

        if True:  # not os.path.exists(fig_fpath):
            model.load(model_fpath)
            model.plot_record(fig_fpath)

        pb_share.update()


def worker(task_all):
    while True:
        task = task_all.get()
        if task is None:
            break
        else:
            train_model_azi(*task)


if __name__ == '__main__':
    #
    fea_dir = 'Data/Features/train'
    record_dir = 'Data/Records/train'
    model_dir = 'models_GMMs_norm_0/all_room'
    azi_all = np.arange(8, 29)

    n_worker = 4
    pb_share = ProcessBarMulti([21*32], desc_all=['train_GMMs'])  # noqa
    task_all = Manager().Queue()
    for azi in range(8, 29):
        train_model_azi(fea_dir, record_dir, azi, model_dir, pb_share)
        # task_all.put((fea_dir, record_dir, azi, model_dir, pb_share))

    # for worker_i in range(n_worker):
    #     task_all.put(None)

    # thread_all = []
    # for worker_i in range(n_worker):
    #     thread = Process(target=worker, args=(task_all, ))
    #     thread.start()
    #     thread_all.append(thread)

    # [thread.join() for thread in thread_all]
