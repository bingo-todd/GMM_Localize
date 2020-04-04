import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

mic_pos_all = [1, 3, 5, 7, 9, 11, 13, 15]


def plot_brirs(brir_dir):
    fig, ax = plt.subplots(1, 1)
    for azi in range(37):
        fpath = f'{brir_dir}/{azi}.mat'
        brir = sio.loadmat(fpath)['data']
        ax.plot(brir)
    ax.set_xlim((0, 400))
    fig_fpath = f'{brir_dir}/brirs.png'
    fig.savefig(fig_fpath)


if __name__ == '__main__':
    for mic_pos in mic_pos_all:
        brir_dir = f'../../Data/BRIRs/train/RT_0.5/{mic_pos}'
        plot_brirs(brir_dir)
