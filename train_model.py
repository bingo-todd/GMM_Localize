import numpy as np
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
import warnings

from BasicTools import wav_tools, plot_tools, get_fpath, ProcessBarMulti
import Auditory_model
import GMMs


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


def file_reader(fea_dir, record_dir, azi, band_i):
    """Load features files, return a samples generator
    """
    fea_fpath_relative_all = get_fpath(fea_dir, suffix='.npz')
    for fea_fpath_relative in fea_fpath_relative_all:
        fea_fpath = os.path.join(fea_dir, fea_fpath_relative)

        *_, reciever_pos, fname = fea_fpath.split('/')
        tar_azi, inter_azi, snr = [int(item) for item in fname[:-4].split('_')]
        if tar_azi != azi:
            continue

        fea_file = np.load(fea_fpath)
        cue_frame_all = fea_file['cue_frame_all']
        ccf_frame_all = fea_file['ccf_frame_all']
        snr_frame_all = fea_file['snr_frame_all']

        n_frame = cue_frame_all.shape[1]

        # feature selection
        # 1. vad, on one channel(L)
        tar_record_fpath = os.path.join(
                    record_dir, '{}_tar.wav'.format(fea_fpath_relative[:-4]))
        tar_record, fs = wav_tools.wav_read(tar_record_fpath)
        inter_record_fpath = os.path.join(
                    record_dir, '{}_inter.wav'.format(fea_fpath_relative[:-4]))
        inter_record, fs = wav_tools.wav_read(inter_record_fpath)

        theta_vad = 40
        vad_flag_frame_all = wav_tools.vad(x=tar_record[:, 0],
                                           frame_len=frame_len,
                                           shift_len=shift_len,
                                           theta=theta_vad, is_plot=False)
        vad_flag_frame_all = vad_flag_frame_all[:n_frame]

        # 2. SNR in each frequency band
        snr_flag_frame_all = snr_frame_all[band_i] > 0.0

        # 3. correlation coefficients
        theta_corr_coef = 0.3
        ccf_flag_frame_all = np.greater(np.max(ccf_frame_all[band_i], axis=1),
                                        theta_corr_coef)

        # 4. ITDs range
        itd_flag_frame_all = np.less(np.abs(cue_frame_all[band_i, :, 0]),
                                     44.0/44.1)  # itd ms

        # combine all criteras
        flag_frame_all = np.logical_and.reduce((vad_flag_frame_all,
                                                snr_flag_frame_all,
                                                ccf_flag_frame_all,
                                                itd_flag_frame_all))

        if False:  # plot waveform and corresponding criteria result
            fs = 44100
            front_end = Auditory_model(fs=fs, cf_low=80, cf_high=5000,
                                       n_band=32, is_middle_ear=True,
                                       ihc_type='Roman')
            t_frame = np.arange(n_frame)*shift_len+int(frame_len/2)
            fig = plt.figure(figsize=(8, 4), tight_layout=True)
            ax1 = plt.subplot(221)
            ax1.plot(np.sum(front_end.filter(inter_record)[band_i], axis=1))
            ax1.plot(np.sum(front_end.filter(tar_record)[band_i], axis=1))
            ax_twin = ax1.twinx()
            ax_twin.plot(t_frame, flag_frame_all, color='red')

            ax2 = plt.subplot(223)
            ax2.plot(t_frame, vad_flag_frame_all+0.09, label='vad')
            ax2.plot(t_frame, snr_flag_frame_all+0.06, label='snr')
            ax2.plot(t_frame, ccf_flag_frame_all+0.03, label='ccf')
            ax2.plot(t_frame, itd_flag_frame_all, label='itd')
            ax2.legend()

            ax3 = plt.subplot(122)
            plot_cue_sample(cue_frame_all[band_i], ax3)
            plot_cue_sample(cue_frame_all[band_i, flag_frame_all, :], ax3)

            plot_tools.savefig(fig, name=f'criterias_eg_{band_i}.png',
                               dir='images/train_model')
            return

        yield cue_frame_all[band_i, flag_frame_all, :]


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
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    n_band = 32
    for band_i in range(n_band):

        model_fpath = os.path.join(model_dir, '{}_{}.npy'.format(azi, band_i))
        fig_fpath = os.path.join(img_dir, '{}_{}.png'.format(azi, band_i))
        gif_fpath = os.path.join(img_dir, '{}_{}.gif'.format(azi, band_i))

        if os.path.exists(model_fpath) and os.path.exists(fig_fpath):
            warnings.warn(f'{model_fpath} already exits')
#            continue

        sample_all = [sample for sample in file_reader(
                                        fea_dir, record_dir, azi, band_i)]

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


def parallel_worker(args):
    func = args[0]
    func(*args[1:])


if __name__ == '__main__':
    #
    fea_dir = 'Data/Features/train'
    record_dir = 'Data/Records/train'
    model_dir = 'models_GMMs_norm_0/all_room'
    azi_all = np.arange(8, 29)

#    train_model_azi(fea_dir, record_dir, 8, model_dir, None)
#    raise Exception()

    if True:
        pb_share = ProcessBarMulti([21*32], desc_all=['train_GMMs'])  # noqa
        arg_all = [(train_model_azi, fea_dir, record_dir,
                    azi_i, model_dir, pb_share)
                   for azi_i in range(8, 29)]
        n_worker = 12
        pool = Pool(n_worker)
        pool.map(parallel_worker, arg_all)

    # plot example of cues screening
    if False:
        azi = 18
        for band_i in range(0, 31, 4):
            file_reader(fea_dir, record_dir, azi, band_i)

    if False:
        azi = 18
        fig, ax = plt.subplots(2, 4, figsize=(12, 4), tight_layout=True,
                               sharex=True, sharey=True)
        for i, band_i in enumerate(range(0, 31, 4)):
            sample_all = [sample for sample in file_reader(fea_dir, record_dir,
                                                           azi, band_i)]
            sample_all = np.concatenate(sample_all, axis=0)
            plot_cue_sample(sample_all, ax[int(i/4), np.mod(i, 4)])
        plot_tools.savefig(fig, name='cue_sample.png',
                           dir='images/train_model')
