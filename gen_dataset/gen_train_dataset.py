import numpy as np
import scipy.io as sio
from multiprocessing import Process, JoinableQueue
import time
import os
from BasicTools import wav_tools, Filter_GPU, get_fpath
from Auditory_model import Auditory_model


TIMIT_dir = '/home/st/Work_Space/Data/TIMIT/44100Hz/TIMIT/TRAIN'
src_fpath_all = [os.path.join(TIMIT_dir, item)
                 for item in get_fpath(TIMIT_dir, '.wav')]

# azimuth difference between target and interfer
azi_diff_all = [-8, -6, -4, -2, -1, 1, 2, 4, 6, 8]
snr_all = [0, 10, 20]

fs = 44100
freq_low = 80
freq_high = 5000

frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)
n_band = 32


def cal_itd_ild(record_tar_fpath, record_inter_fpath, fea_fpath, front_end):

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


def get_src_fpath(n):
    return np.random.choice(src_fpath_all, size=n, replace=False)


def gen_set_pos(room, mic_pos, filter_gpu, front_end):
    """"Generate dataset for the given room and the given reciever position
    """
    record_dir = '../Data/Records/train/{}/{}'.format(room, mic_pos)
    fea_dir = '../Data/Features/train/{}/{}'.format(room, mic_pos)
    # record source
    src_fpath_file = open(os.path.join(record_dir, 'src_fpath.txt'), mode='a')
    date_str_temp = ''.join(('Time: {0.tm_year}-{0.tm_mon}-{0.tm_mday}-',
                             '{0.tm_hour}-{0.tm_min}-{0.tm_sec}\n'))
    src_fpath_file.write(date_str_temp.format(time.localtime()))
    brir_fpath_temp = '../Data/BRIRs/train/{}/{}/{}.mat'.format(
                                                    room, mic_pos, '{}')
    for azi_tar in np.arange(8, 29):  # [-50,50]
        brir_tar = sio.loadmat(brir_fpath_temp.format(azi_tar))['data']
        for azi_diff in azi_diff_all:
            azi_inter = azi_tar+azi_diff
            brir_inter = sio.loadmat(brir_fpath_temp.format(azi_inter))['data']
            for snr in snr_all:
                src_tar_fpath, src_inter_fpath = get_src_fpath(2)
                if src_tar_fpath == src_inter_fpath:
                    raise Exception()
                # record sourc of target and interfer
                # src_fpath_file.write('{}_{}_{} {} {}\n'.format(
                #                                 azi_tar, azi_inter,
                #                                 snr, src_tar_fpath,
                #                                 src_inter_fpath))
                # src_fpath_file.flush()

                record_tar_fpath = os.path.join(record_dir,
                                                '{}_{}_{}_tar.wav'.format(
                                                    azi_tar, azi_inter, snr))
                record_inter_fpath = os.path.join(record_dir,
                                                  '{}_{}_{}_inter.wav'.format(
                                                    azi_tar, azi_inter, snr))

                fea_fpath = os.path.join(fea_dir,
                                         '{}_{}_{}.npz'.format(
                                                    azi_tar, azi_inter, snr))

                # if (os.path.exists(record_inter_fpath) and
                #     os.path.exists(record_inter_fpath) and
                #     os.path.exists(fea_fpath)):
                #     continue
                # else:
                #     print(fea_fpath)
                #     raise Exception()

                # spatializing
                # target
                src_tar, fs = wav_tools.wav_read(src_tar_fpath)
                record_tar = filter_gpu.brir_filter(src_tar, brir_tar)
                wav_tools.write_wav(record_tar, fs, record_inter_fpath)
                # interference
                src_inter, fs = wav_tools.wav_read(src_inter_fpath)
                src_inter = wav_tools.set_snr(src_inter, src_tar, -snr)
                record_inter = filter_gpu.brir_filter(src_inter, brir_inter)
                wav_tools.write_wav(record_inter, fs, record_inter_fpath)

                cal_itd_ild(record_tar_fpath, record_inter_fpath,
                            fea_fpath, front_end)

                # fig,ax = plt.subplots(2,2)
                # ax[0,0].plot(src_tar);ax[0,1].plot(record_tar)
                # ax[1,0].plot(src_inter);ax[1,1].plot(record_inter)
                # ax[1,0].set_title('{}'.format(snr))
                # fig.savefig('test.png')
                # raise Exception()

    src_fpath_file.close()


def parallel_worker(func, arg_all):
    filter_gpu = Filter_GPU.Filter_GPU(gpu_index=0)
    front_end = Auditory_model(fs=fs, freq_low=freq_low, freq_high=freq_high,
                               n_band=32, is_middle_ear=True,
                               ihc_type='Roman')
    while True:
        arg = arg_all.get_nowait()
        if arg is None:
            arg_all.task_done()
            break

        func(*arg, filter_gpu, front_end)
        arg_all.task_done()


def gen_train_set(n_worker=32):
    # fixed room environment, RT60=0.5
    room_all = ['RT_0.5']
    mic_pos_all = [1, 3, 5, 7, 9, 11, 13, 15]

    arg_all = JoinableQueue()
    for room in room_all:
        for mic_pos in mic_pos_all:
            arg_all.put((room, mic_pos))

    for i in range(n_worker):
        arg_all.put(None)

    thread_all = []
    for i in range(n_worker):
        thread = Process(target=parallel_worker, args=(gen_set_pos, arg_all))
        thread.start()
        thread_all.append(thread)
    [thread.join() for thread in thread_all]


# def gen_valid_set(n_worker=32):
#    # fixed room environment, RT60=0.5
#    room_all = ['RT_0.5']
#    reciever_pos_all = [1,3,5,7,9,11,13,15]
#
#    arg_all = queue.Queue()
#    for room in room_all:
#        for reciever_pos in reciever_pos_all:
#            record_dir = 'Data1/Records/train/{}/{}'.format(room,reciever_pos)
#            if not os.path.exists(record_dir):
#                os.makedirs(record_dir)
#            brir_fpath_temp = 'Data/BRIRs/train/{}/{}/{}.mat'.format(
#                                                    room,reciever_pos,'{}')
#            arg_all.put((record_dir,brir_fpath_temp))
#
#    thread_all = []
#    for i in range(n_worker):
#        thread = Thread(target=parallel_worker,args=(gen_set_pos,arg_all),
#                        daemon=True)
#        thread.start()
#        thread_all.append(thread)
#
#    [thread.join() for thread in thread_all]


if __name__ == '__main__':
    #
    gen_train_set()
