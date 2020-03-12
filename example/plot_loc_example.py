import numpy as np
import os
import sys
sys.path.append('../')
from GMM_Localizer import GMM_Localizer  # noqa: E402
sys.path.append('../utils')
from Plot_Scene import Plot_Scene  # noqa: E402

# constant variables

fs = 44100
n_band = 32
freq_low = 80
freq_high = 5000
frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)
azi_theta = 1

room_all = ['RT_0.19', 'RT_0.29', 'RT_0.39', 'RT_0.48', 'RT_0.58', 'RT_0.69']
mic_pos_test_all = [2, 4, 5, 6, 8, 10, 12, 14]
azi_tar_all = np.arange(8, 29)
n_inter_all = [0, 1, 2, 3]

n_test = 3
n_room = len(room_all)
n_mic_pos = len(mic_pos_test_all)
n_azi_tar = azi_tar_all.shape[0]

test_set_dir = '../Data/Features/test/'


def load_test_sample(room, mic_pos, azi_tar, n_inter, test_i):

    cue_fpath = os.path.join(test_set_dir, room, f'{mic_pos}',
                             f'{azi_tar}_{n_inter}_{test_i}.npy')
    [cue_frame_all,
     ccf_frame_all,
     src_azi_all] = np.load(cue_fpath, allow_pickle=True)
    return cue_frame_all, src_azi_all


def plot_result(room, mic_pos, azi_tar, n_inter, model):

    test_i = 0

    cues, azi_gt_all = load_test_sample(room, mic_pos,
                                        azi_tar, n_inter, test_i)

    azi_est = model.locate(cues)

    scene = Plot_Scene(src_azi_all=azi_gt_all, mic_pos=mic_pos, room=room)
    scene.plot_est(azi_est)
    scene.savefig(os.path.join('../images/example/loc_example/',
                               '{}_{}_{}_{}_scence.png'.format(
                                        room, mic_pos, azi_tar, n_inter)))

    # fig,ax = plt.subplots(3,1,sharex=True,tight_layout=True)
    # for record_i,record in enumerate(record_all):
    #     ax[0].plot(np.arange(record.shape[0])/fs,record[:,0],
    #                label='source {}'.format(record_i))
    #     ax[1].plot(np.arange(record.shape[0])/fs,record[:,1],
    #                label='source {}'.format(record_i))
    # ax[0].legend()
    # ax[0].set_title('L')
    # ax[1].legend()
    # ax[1].set_title('R')
    # ax[2].plot(np.arange(n_frame)*shift_len+int(frame_len/2),azi_est)
    # for azi_gt_i,azi_gt in enumerate(azi_gt_all):
    #     ax[2].plot([0,(n_frame-1)*shift_len+int(frame_len/2)],[azi_gt,azi_gt],
    #                label='source {}'.format(azi_gt_i))
    # ax[2].legend()
    # ax[2].set_ylim([7,29])
    # ax[2].set_title('{:.2f}'.format(cp))
    # plot_tools.savefig(fig,name='{}_{}_{}_{}_result.png'.format(room,mic_pos,
    #                                                             azi_tar,n_inter))


if __name__ == '__main__':

    model = GMM_Localizer(model_dir='../models/all_room')

    mic_pos = 5
    # for room in room_all:
    room = 'RT_0.29'
    for azi_tar in range(8, 29, 5):
        for n_inter in n_inter_all:
            plot_result(room, mic_pos, azi_tar, n_inter, model)
