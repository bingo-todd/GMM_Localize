import matplotlib.pyplot as plt
import numpy as np
from BasicTools import plot_tools

fs = 44100
n_band = 32
freq_low = 80
freq_high = 5000
frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)
azi_diff_theta = 2

room_all = ['RT_0.19', 'RT_0.29', 'RT_0.39', 'RT_0.48', 'RT_0.58', 'RT_0.69']
rt_all = [np.float32(room[3:]) for room in room_all]
mic_pos_all = [2, 4, 5, 6, 8, 10, 12, 14]
azi_tar_all = np.arange(8, 29)
n_inter_all = [0, 1, 2, 3]

n_test = 3
n_room = len(room_all)
n_mic_pos = len(mic_pos_all)
n_azi_tar = azi_tar_all.shape[0]
n_n_inter = len(n_inter_all)


def plot_result():
    cp_all = np.zeros((n_room, n_mic_pos, n_azi_tar, n_n_inter, n_test))
    rmse_all = np.zeros((n_room, n_mic_pos, n_azi_tar, n_n_inter, n_test))
    for room_i, room in enumerate(room_all):
        for mic_pos_i, mic_pos in enumerate(mic_pos_all):
            result_fpath = '../result/{}_{}.npy'.format(room, mic_pos)
            [cp_all[room_i, mic_pos_i],
             rmse_all[room_i, mic_pos_i]] = np.load(result_fpath)

    fig, ax = plt.subplots(1, 2, figsize=[8, 3], sharex=True,
                           tight_layout=True)
    for n_inter_i in range(n_n_inter):
        ax[0].plot(range(n_room), np.mean(rmse_all[:, :, :, n_inter_i, :],
                                          axis=(1, 2, 3)),
                   alpha=0.5, label=f'{n_inter_i+1} src')
        ax[1].plot(range(n_room), 100*(1-np.mean(cp_all[:, :, :, n_inter_i, :],
                                                 axis=(1, 2, 3))),
                   alpha=0.5, label=f'{n_inter_i+1} src')

    ax[0].plot(range(n_room), np.mean(rmse_all, axis=(1, 2, 3, 4)),
               alpha=1, color='black', label=f'mean')
    ax[0].xaxis.set_ticks(range(n_room))
    ax[0].set_xticklabels([room[3:] for room in room_all])
    ax[0].set_xlabel('Room')
    ax[0].set_ylabel('RMSE($^o$)')
    # ax[0].spines['top'].set_visible(False)

    ax[1].plot(range(n_room), 100*(1-np.mean(cp_all, axis=(1, 2, 3, 4))),
               alpha=1, color='black', label=f'mean')
    ax[1].legend()
    ax[1].set_xlabel('Room')
    ax[1].set_ylim([0, 60])
    ax[1].set_ylabel('Anomalies(%)')
    plot_tools.savefig(fig, fig_name='result_room.png',
                       fig_dir='../images/result')


if __name__ == '__main__':
    plot_result()
