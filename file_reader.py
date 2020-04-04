import numpy as np
import matplotlib.pyplot as plt
import os
from BasicTools import get_fpath, wav_tools, plot_tools, ProcessBar
from Auditory_model import Auditory_model


fs = 16000
freq_low = 80
freq_high = 5000

frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)
n_band = 32

snr_all = [0, 10, 20]
azi_diff_all = [-8, -6, -4, -2, -1, 1, 2, 4, 6, 8]
room_all = ['RT_0.5']
mic_pos_all = [1, 3, 5, 7, 9, 11, 13, 15]
azi_min = 9
azi_max = 27


def plot_cue_sample(cues, ax):
    # fig,ax = plt.subplots(1,0)
    ax.scatter(cues[:, 0], cues[:, 1], alpha=0.3)
    ax.set_xlabel('ITD(ms)')
    ax.set_ylabel('ILD(dB)')
    # return fig


def load_src_fpath(record_dir):
    src_fpath_fpath_all = get_fpath(record_dir, suffix='.txt',
                                    is_absolute=True)
    fpath_all = {}
    for src_fpath_fpath in src_fpath_fpath_all:
        *_, room, mic_pos, fname = src_fpath_fpath.split('/')
        if fname != 'src_fpath.txt':
            raise Exception
        if room not in fpath_all.keys():
            fpath_all[room] = {}
        if mic_pos not in fpath_all[room].keys():
            fpath_all[room][mic_pos] = {}
        with open(src_fpath_fpath, 'r') as src_fpath_file:
            line_all = src_fpath_file.readlines()
            for line in line_all[1:]:
                condition, fpath_tar, fpath_inter = line.split()
                # print(condition, fpath_tar, fpath_inter)
                fpath_all[room][mic_pos][condition] = [fpath_tar, fpath_inter]
    return fpath_all


def file_reader(fea_dir, band_tar=None, azi_tar=None,
                is_screen=False, record_dir=None,
                is_plot=False, fig_name=None, is_pb=False):
    #
    theta_vad = 40
    theta_corr_coef = 0.3
    theta_itd = 44.0/44.1

    if is_screen:
        src_fpath_all = load_src_fpath(record_dir)

    fea_fpath_all = get_fpath(fea_dir, suffix='.npz', is_absolute=True)
    pb = ProcessBar(len(fea_fpath_all))
    for fea_fpath in fea_fpath_all:
        if is_pb:
            pb.update()
        *_, room, mic_pos, fname = fea_fpath[:-4].split('/')
        azi, wav_i, snr = [np.int16(item) for item in fname.split('_')]
        if (azi_tar is not None) and (azi != azi_tar):
            continue

        fea_file = np.load(fea_fpath)
        cue_frame_all = fea_file['cue_frame_all']
        ccf_frame_all = fea_file['ccf_frame_all']
        snr_frame_all = fea_file['snr_frame_all']

        if not is_screen:
            if band_tar is None:
                yield np.transpose(cue_frame_all, axes=(1, 0, 2))
            else:
                yield cue_frame_all[band_tar]
        else:
            n_frame = cue_frame_all.shape[1]
            flag_frame_all_band_all = []

            # feature selection
            # 1. vad, on one channel(L)
            src_fpath_tar = \
                src_fpath_all[room][mic_pos][f'{azi}_{wav_i}_{snr}'][0]
            src_fpath_tar = src_fpath_tar.replace('Data/TIMIT',
                                                  'Data/TIMIT_wav')
            src_tar, fs = wav_tools.read_wav(src_fpath_tar)

            tar_fpath = ''.join((f'{record_dir}/{room}/{mic_pos}/',
                                 f'{azi}_{wav_i}_{snr}_tar.wav'))
            tar, fs = wav_tools.read_wav(tar_fpath)
            if tar.shape[0] != src_tar.shape[0]:
                raise Exception()

            # time delay between source and recording, about 70 samples
            delay = 190
            src_tar = np.concatenate((src_tar[delay:], np.zeros(delay)))
            vad_flag_frame_all = wav_tools.vad(x=src_tar,
                                               frame_len=frame_len,
                                               shift_len=shift_len,
                                               theta=theta_vad, is_plot=False)
            vad_flag_frame_all = np.squeeze(vad_flag_frame_all[:n_frame])

            for band_i in range(n_band):
                if band_tar is not None:
                    if band_i != band_tar:
                        continue

                # 2. SNR in each frequency band
                snr_flag_frame_all = snr_frame_all[band_i] > 0.0

                # 3. correlation coefficients
                ccf_flag_frame_all = np.greater(
                                        np.max(ccf_frame_all[band_i], axis=1),
                                        theta_corr_coef)

                # 4. ITDs range
                itd_flag_frame_all = np.less(
                                        np.abs(cue_frame_all[band_i, :, 0]),
                                        theta_itd)  # itd ms

                # combine all criteras
                flag_frame_all = np.logical_and.reduce((vad_flag_frame_all,
                                                        snr_flag_frame_all,
                                                        ccf_flag_frame_all,
                                                        itd_flag_frame_all))
                flag_frame_all_band_all.append(flag_frame_all)

            # plot waveform and corresponding criteria result
            if is_plot:
                tar_fpath = os.path.join('{}_{}.wav'.format(azi, wav_i))
                tar, fs = wav_tools.read_wav(tar_fpath)

                inter_fpath = '{}_{}_{}.wav'.format(azi, wav_i, snr)
                inter, fs = wav_tools.read_wav(inter_fpath)

                front_end = Auditory_model(fs=fs, cf_low=freq_low,
                                           cf_high=freq_high, n_band=n_band,
                                           is_middle_ear=True,
                                           ihc_type='Roman')
                t_frame = np.arange(n_frame)*shift_len+int(frame_len/2)
                fig = plt.figure(figsize=(8, 4), tight_layout=True)
                ax1 = plt.subplot(221)
                ax1.plot(np.sum(front_end.filter(inter)[band_i], axis=1))
                ax1.plot(np.sum(front_end.filter(tar)[band_i], axis=1))
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

                plot_tools.savefig(fig, fig_name=fig_name, fig_dir='./')
                return

            if band_i is None:
                flag_frame_all = np.logical_and.reduce(flag_frame_all_band_all)
                yield np.transpose(cue_frame_all[:, flag_frame_all, :],
                                   axes=(1, 0, 2))
            else:
                flag_frame_all = flag_frame_all_band_all[0]
                yield cue_frame_all[band_tar, flag_frame_all, :]


if __name__ == '__main__':

    # fpath_all = load_src_fpath('Data/Records/train')
    # file_reader('Data/Features/train', band_tar=20, is_screen=True,
    #             record_dir='Data/Records/train')

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(4, 6),
                           tight_layout=True)

    sample_generator = file_reader(
                fea_dir='Data/Features/train/', band_tar=30, is_screen=True,
                record_dir='Data/Records/train',
                azi_tar=9)
    sample_all = np.concatenate([sample for sample in sample_generator],
                                axis=0)
    plot_cue_sample(sample_all, ax=ax[0])
    ax[0].set_title('9')

    sample_generator = file_reader(
                fea_dir='Data/Features/train/', band_tar=30, is_screen=True,
                record_dir='Data/Records/train',
                azi_tar=27)
    sample_all = np.concatenate([sample for sample in sample_generator],
                                axis=0)
    plot_cue_sample(sample_all, ax=ax[1])
    ax[1].set_title('27')

    fig.savefig('cues_azi_9_27_band_30.png')
