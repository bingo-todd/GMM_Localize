import numpy as np
import scipy.io as sio
from multiprocessing import Process, JoinableQueue
import os
import sys

from BasicTools import wav_tools, ProcessBarMulti, get_fpath, Filter_GPU
from Auditory_model import Auditory_model


# constant variables

fs = 44100
n_band = 32
freq_low = 80
freq_high = 5000
frame_len = int(fs*20e-3)
shift_len = int(fs*10e-3)
max_delay = int(fs*1e-3)
azi_sep_theta = 2


room_all = ['RT_0.19', 'RT_0.29', 'RT_0.39', 'RT_0.48',
            'RT_0.58', 'RT_0.69']
mic_pos_test_all = [2, 4, 5, 6, 8, 10, 12, 14]
azi_tar_all = np.arange(8, 29)
n_inter_all = [0, 1, 2, 3]

n_test = 3
n_room = len(room_all)
n_mic_pos = len(mic_pos_test_all)
n_azi_tar = azi_tar_all.shape[0]
n_n_inter = len(n_inter_all)


record_set_dir = '../Data/Records/test'
fea_set_dir = '../Data/Features/test'


# wave file path of TIMIT test set
TIMIT_dir = '/home/st/Work_Space/Data/TIMIT/44100Hz/TIMIT/TEST'
src_fpath_all = [os.path.join(TIMIT_dir, item)
                 for item in get_fpath(TIMIT_dir, '.wav')]


def get_wav_fpath(n):
    return np.random.choice(src_fpath_all, size=n, replace=False)


def syn_record(src, room, mic_pos, azi, filter_gpu):
    brir_fpath = '../Data/BRIRs/test/{}/{}/{}.mat'.format(room, mic_pos, azi)
    brir = sio.loadmat(brir_fpath)['data']
    return filter_gpu.brir_filter(src, brir)


def gen_test_sample(room, mic_pos, azi_tar, n_inter, test_i,
                    filter_gpu, front_end):

    src_azi_all = np.zeros(n_inter+1)
    src_azi_all[0] = azi_tar

    src_fpath_all = get_wav_fpath(n_inter+1)
    src_tar, _ = wav_tools.read_wav(src_fpath_all[0])
    record_tar = syn_record(src_tar, room, mic_pos, azi_tar, filter_gpu)

    mix = record_tar
    mix_len = mix.shape[0]
    for i in range(n_inter):
        # minimal azimuth separation 10^o
        inter_azi = azi_tar
        while np.abs(azi_tar-inter_azi) < azi_sep_theta:
            inter_azi = np.random.choice(azi_tar_all, size=1)[0]
        src_azi_all[i+1] = inter_azi

        src_inter, _ = wav_tools.read_wav(src_fpath_all[1+i])
        src_inter_norm = wav_tools.set_snr(src_inter, src_tar, 0)
        record_inter = syn_record(src_inter_norm, room, mic_pos,
                                  inter_azi, filter_gpu)
        mix_len = min((mix_len, record_inter.shape[0]))
        mix = mix[:mix_len] + record_inter[:mix_len]

    mix_fpath = os.path.join(record_set_dir, room, f'{mic_pos}',
                             '_'.join((f'{azi_tar}', f'{n_inter}',
                                       f'{test_i}.npy')))
    os.makedirs(os.path.dirname(mix_fpath), exist_ok=True)
    np.save(mix_fpath, [mix, src_azi_all])

    fea_fpath = os.path.join(fea_set_dir, room, f'{mic_pos}',
                             '_'.join((f'{azi_tar}', f'{n_inter}',
                                       f'{test_i}.npy')))
    os.makedirs(os.path.dirname(fea_fpath), exist_ok=True)

    [cue_frame_all,
     ccf_frame_all] = front_end.cal_cues(tar=mix, frame_len=frame_len,
                                         shift_len=shift_len,
                                         max_delay=max_delay, n_worker=1)
    np.save(fea_fpath, [cue_frame_all, ccf_frame_all, src_azi_all])


def parallel_worker(arg_all, pb_share):
    filter_gpu = Filter_GPU.Filter_GPU(0)
    front_end = Auditory_model(fs=fs, freq_low=freq_low, freq_high=freq_high,
                               n_band=32, is_middle_ear=True,
                               ihc_type='Roman')
    while True:
        arg = arg_all.get()
        if arg is None:
            arg_all.task_done()
            break
        gen_test_sample(*arg, filter_gpu, front_end)
        arg_all.task_done()
        pb_share.update()


if __name__ == '__main__':
    n_worker = int(sys.argv[1])

    arg_all = JoinableQueue()
    n_count = 0
    for room in room_all:
        for mic_pos in mic_pos_test_all:
            for azi_tar in azi_tar_all:
                for n_inter in range(n_n_inter):
                    for test_i in range(n_test):
                        arg_all.put((room, mic_pos, azi_tar, n_inter, test_i))
                        n_count = n_count + 1

    for i in range(n_worker):
        arg_all.put(None)

    pb_share = ProcessBarMulti([n_count], desc_all=['gen test set']) # noqa: W0401
    proc_all = []
    for i in range(n_worker):
        proc = Process(target=parallel_worker,
                       args=(arg_all, pb_share))
        proc.start()
        proc_all.append(proc)

    arg_all.join()
