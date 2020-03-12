import numpy as np
import matplotlib.pyplot as plt

from BasicTools import wav_tools, plot_tools
from Auditory_model import Auditory_model


# constant variables
fs = 44100
n_band = 32
frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)


def plot_cue_sample(cues, ax, label=None):
    ax.scatter(cues[:, 0], cues[:, 1], alpha=0.3, label=label)
    ax.set_xlabel('ITD(ms)')
    ax.set_ylabel('ILD(dB)')


if __name__ == '__main__':
    """Load features files, return a samples generator
    """

    fea_fpath = '../Data/Features/train/RT_0.5/5/19_11_20.npz'
    tar_fpath = '../Data/Records/train/RT_0.5/5/19_11_20_tar.wav'
    inter_fpath = '../Data/Records/train/RT_0.5/5/19_11_20_inter.wav'
    band_i = 20

    *_, mic_pos, fname = fea_fpath.split('/')
    tar_azi, inter_azi, snr = [int(item) for item in fname[:-4].split('_')]

    fea_file = np.load(fea_fpath)
    cue_frame_all = fea_file['cue_frame_all']
    ccf_frame_all = fea_file['ccf_frame_all']
    snr_frame_all = fea_file['snr_frame_all']

    n_frame = cue_frame_all.shape[1]

    # 1. vad, on one channel(L)
    tar_record, fs = wav_tools.read_wav(tar_fpath)
    inter_record, fs = wav_tools.read_wav(inter_fpath)

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

    fs = 44100
    front_end = Auditory_model(fs=fs, cf_low=80, cf_high=5000,
                               n_band=32, is_middle_ear=True,
                               ihc_type='Roman')
    t_frame = np.arange(n_frame)*shift_len+int(frame_len/2)
    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    ax1 = plt.subplot(221)
    ax1.plot(t_frame, flag_frame_all, color='red', label='Criteria')
    ax1.legend(loc='upper right')
    ax_twin = ax1.twinx()
    ax_twin.plot(np.sum(front_end.filter(tar_record)[band_i], axis=1),
                 label='tar')
    ax_twin.plot(np.sum(front_end.filter(inter_record)[band_i], axis=1),
                 label='inter')
    ax_twin.legend(loc='lower right')

    ax2 = plt.subplot(223)
    ax2.plot(t_frame, vad_flag_frame_all+0.09, label='vad')
    ax2.plot(t_frame, snr_flag_frame_all+0.06, label='snr')
    ax2.plot(t_frame, ccf_flag_frame_all+0.03, label='ccf')
    ax2.plot(t_frame, itd_flag_frame_all, label='itd')
    ax2.legend()

    ax3 = plt.subplot(122)
    plot_cue_sample(cue_frame_all[band_i], ax3, label='all')
    plot_cue_sample(cue_frame_all[band_i, flag_frame_all, :], ax3,
                    label='preserve')
    ax3.legend()
    plot_tools.savefig(fig, fig_name=f'criterias_eg_{band_i}.png',
                       fig_dir='../images/train')
