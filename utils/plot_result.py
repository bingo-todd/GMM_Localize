import numpy as np
import matplotlib.pyplot as plt
import os
import sys

my_modules_dir = os.path.join(os.path.expanduser('~'), 'my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import plot_tools  # noqa
from ProcessBar import ProcessBar  # noqa
# from Plot_Scene import Plot_Scene


room_all = ['RT_0.19', 'RT_0.29', 'RT_0.39', 'RT_0.48', 'RT_0.58', 'RT_0.69']
RT_all = [0.19, 0.29, 0.39, 0.48, 0.58, 0.69]
n_room = len(room_all)
mic_label_all = [2, 4, 5, 6, 8, 10, 12, 14]
n_mic_pos = len(mic_label_all)
tar_azi_all = np.arange(8, 29)
n_azi_tar = tar_azi_all.shape[0]

n_inter_all = [0]#, 1, 2, 3]
n_n_inter = len(n_inter_all)
n_test = 3


def cal_result(azi_theta):

    result_dir = 'Result_raw/all_room'
    rmse_all = np.zeros((n_room, n_mic_pos, n_azi_tar, n_n_inter, n_test))
    cp_all = np.zeros((n_room, n_mic_pos, n_azi_tar, n_n_inter, n_test))

    pb = ProcessBar(n_room*n_mic_pos*n_azi_tar*n_n_inter*n_test)

    fig, ax = plt.subplots(1, 1)

    for room_i, room in enumerate(room_all):
        for mic_label_i, mic_label in enumerate(mic_label_all):
            for azi_tar_i, azi_tar in enumerate(tar_azi_all):
                for n_inter_i, n_inter in enumerate(n_inter_all):
                    for test_i in range(n_test):
                        pb.update()

                        raw_result_fpath = os.path.join(
                            result_dir,
                            '_'.join((f'{room}', f'{mic_label}', f'{azi_tar}',
                                      f'{n_inter}', f'{test_i}.npz')))
                        raw_result = np.load(raw_result_fpath)
                        azi_est = raw_result['azi_est']
                        azi_gt_all = raw_result['azi_gt_all']

                        azi_diff = np.min(
                                    np.abs(
                                        np.subtract(
                                            azi_gt_all.reshape([-1, 1]),
                                            azi_est)),
                                    axis=0)
                        n_frame = azi_est.shape[0]
                        n_frame_correct = np.count_nonzero(
                                                azi_diff <= azi_theta)
                        cp_tmp = np.float32(n_frame_correct)/n_frame

                        cp_all[room_i, mic_label_i,
                               azi_tar_i, n_inter_i, test_i] = cp_tmp

                        rmse_all[room_i, mic_label_i,
                                 azi_tar_i, n_inter_i,
                                 test_i] = np.sqrt(np.mean(azi_diff**2))*5

                        # ax.plot(azi_est)
                        # for azi_gt in azi_gt_all:
                        #     ax.plot([0, azi_est.shape[0]], [azi_gt, azi_gt])
                        # ax.set_ylim([7,30])
                        # ax.set_title(f'{cp_tmp}')
                        # plt.show(block=False)
                        # print(azi_gt_all)
                        # print(azi_diff)
                        # input()

                        ax.clear()

    return [cp_all, rmse_all]


def plot_result(cp_all, rmse_all,fig_dir):

    # rmse
    fig_rmse,ax_rmse = plt.subplots(1,1)
    fig_cp,ax_cp = plt.subplots(1,1)
    for n_inter_i in range(n_n_inter):
        rmse_all_tmp = np.mean(rmse_all[:,:,:,n_inter_i,:],axis=(1,2,3))
        ax_rmse.plot(RT_all,rmse_all_tmp,alpha=0.5,linewidth=2,label=f'{n_inter_all[n_inter_i]}+1')

        cp_all_tmp = 100*(1-np.mean(cp_all[:,:,:,n_inter_i,:],axis=(1,2,3)))
        ax_cp.plot(RT_all,cp_all_tmp,alpha=0.5,linewidth=2,label=f'{n_inter_all[n_inter_i]+1}')

    ax_rmse.plot(RT_all,np.mean(rmse_all,axis=(1,2,3,4)),alpha=1,color='black',linewidth=2,label='Mean')
    ax_cp.plot(RT_all,100*(1-np.mean(cp_all,axis=(1,2,3,4))),alpha=1,color='black',linewidth=2,label='Mean')

    ax_rmse.set_ylim((0,25))
    ax_rmse.set_xlabel('RT')
    ax_rmse.set_ylabel('RMSE[$^o$]')

    ax_cp.set_xlabel('RT')
    ax_cp.set_ylim([0,60])
    ax_cp.set_ylabel('Anomalies[%]')
    ax_cp.legend()
    # ax[0].xaxis.set_major_locator(plt.MultipleLocator(1))
    # ax[0].set_xticklabels()
    plot_tools.savefig(fig_rmse, 'rmse_result_all.png', fig_dir)
    plot_tools.savefig(fig_cp, 'cp_result_all.png', fig_dir)



def plot_result_room(cp_all, rmse_all,fig_dir):

    # rmse
    tar_azi_all = np.arange(8,29)
    for room_i,room in enumerate(room_all):
        fig_rmse,ax_rmse = plt.subplots(1,1)
        fig_cp,ax_cp = plt.subplots(1,1)
        for n_inter_i in range(n_n_inter):
            rmse_all_tmp = np.mean(rmse_all[room_i,:,:,n_inter_i,:],axis=(0,2))
            ax_rmse.plot(tar_azi_all,rmse_all_tmp,alpha=0.5,linewidth=2,label=f'{n_inter_all[n_inter_i]}+1')

            cp_all_tmp = 100*(1-np.mean(cp_all[room_i,:,:,n_inter_i,:],axis=(0,2)))
            ax_cp.plot(tar_azi_all,cp_all_tmp,alpha=0.5,linewidth=2,label=f'{n_inter_all[n_inter_i]+1}')

        ax_rmse.plot(tar_azi_all,np.mean(rmse_all[room_i],axis=(0,2,3)),alpha=1,color='black',linewidth=2,label='Mean')
        ax_cp.plot(tar_azi_all,100*(1-np.mean(cp_all[room_i],axis=(0,2,3))),alpha=1,color='black',linewidth=2,label='Mean')

        ax_rmse.set_ylim((0,25))
        ax_rmse.set_xlabel('azi')
        ax_rmse.set_ylabel('RMSE[$^o$]')

        ax_cp.set_xlabel('azi')
        ax_cp.set_ylim([0,60])
        ax_cp.set_ylabel('Anomalies[%]')
        ax_cp.legend()
        # ax[0].xaxis.set_major_locator(plt.MultipleLocator(1))
        # ax[0].set_xticklabels()
        plot_tools.savefig(fig_rmse, f'rmse_result_all_{room}.png', fig_dir)
        plot_tools.savefig(fig_cp, f'cp_result_all_{room}.png', fig_dir)



if __name__ == '__main__':
    cp_all, rmse_all = cal_result(azi_theta=1)
    np.savez('Result/all_room/result_all.npz',cp_all=cp_all,rmse_all=rmse_all)

    # result_info = np.load('Result/all_room/result_all.npz')
    # cp_all = result_info['cp_all']
    # rmse_all = result_info['rmse_all']

    plot_result(cp_all,rmse_all,'images/result/all_room/inter_0')
    plot_result_room(cp_all,rmse_all,'images/result/all_room/_inter_0')
