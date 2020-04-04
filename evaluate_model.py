import numpy as np
import os
from multiprocessing import Pool
from GMM_Localizer import GMM_Localizer
from BasicTools import ProcessBar
# constant variables

fs = 44100
n_band = 32
freq_low = 80
freq_high = 5000
frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)
azi_diff_theta = 2

room_all = ['RT_0.19', 'RT_0.29', 'RT_0.39', 'RT_0.48', 'RT_0.58', 'RT_0.69']
mic_pos_test_all = [2, 4, 5, 6, 8, 10, 12, 14]
azi_tar_all = np.arange(8, 29)
n_inter_all = [0, 1, 2, 3]

n_test = 3
n_room = len(room_all)
n_mic_pos = len(mic_pos_test_all)
n_azi_tar = azi_tar_all.shape[0]
n_n_inter = len(n_inter_all)


def evaluate_mic(args):
    model, room, mic_pos = args
    print(f'{room}  {mic_pos}')
    result_fpath = f'{room}_{mic_pos}.npy'
    if os.path.exists(result_fpath):
        return

    cp_all = np.zeros((n_azi_tar, n_n_inter, n_test))
    rmse_all = np.zeros((n_azi_tar, n_n_inter, n_test))
    # pb = ProcessBar(len(azi_tar_all)*len(n_inter_all)*n_test,
    #                 title=f'{room}_{mic_pos}')
    for azi_tar_i, azi_tar in enumerate(azi_tar_all):
        for n_inter_i, n_inter in enumerate(n_inter_all):
            for test_i in range(n_test):
                # pb.update()
                cue_fpath = ''.join((f'Data/Features/test/{room}/{mic_pos}/',
                                    f'{azi_tar}_{n_inter}_{test_i}.npy'))
                cues, ccf, azi_gt_all = np.load(cue_fpath, allow_pickle=True)
                azi_est = model.locate(cues)
                azi_diff = np.min(
                                np.abs(
                                    np.subtract(
                                        azi_gt_all.reshape([-1, 1]),
                                        azi_est)),
                                axis=0)
                n_frame = azi_est.shape[0]
                n_frame_correct = np.count_nonzero(azi_diff <= azi_diff_theta)
                cp_all[azi_tar_i,
                       n_inter_i,
                       test_i] = np.float32(n_frame_correct) / n_frame

                rmse_all[azi_tar_i,
                         n_inter_i,
                         test_i] = np.sqrt(np.mean(azi_diff**2))

    result_fpath = f'result/{room}_{mic_pos}.npy'
    np.save(result_fpath, [cp_all, rmse_all])


def evaluate_main():
    model_dir = 'models/all_room'
    model = GMM_Localizer(model_dir=model_dir)

    pool = Pool(6)
    pool.map(evaluate_mic, [(model, room, mic_pos)
                            for room in room_all
                            for mic_pos in mic_pos_test_all])
    # for room in room_all:
    #     for mic_pos in mic_pos_test_all:
    #         cp_all, rmse_all = evaluate_mic(model, room, mic_pos)
    #         np.save(result_fpath_tmp, [cp_all, rmse_all])


if __name__ == '__main__':
    evaluate_main()
