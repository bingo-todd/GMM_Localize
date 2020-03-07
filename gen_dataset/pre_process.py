import os
import shutil
#



def fname_minus1(dir):
    for azi in range(36):
        shutil.move(os.path.join(dir,'{}.mat'.format(azi+1)),
                    os.path.join(dir,'{}.mat'.format(azi)))

if __name__ == '__main__':

    # format the file path of brir base the template, room/reciever_pos/azi.mat
    if False: # train set
        room_all = ['RT_0.5']
        reciever_pos_all = [1,3,5,7,9,11,13,15]
        for room in room_all:
            for reciever_pos in reciever_pos_all:
                dir = 'BRIRs/train/{}/{}'.format(room,reciever_pos)
                os.makedirs(dir)
                for azi in range(37):
                    src_fpath = 'BRIRs_GMM_backup/train/pos_{}_S{}.mat'.format(reciever_pos,azi+1)
                    dest_fpath = os.path.join(dir,'{}.mat'.format(azi))
                    shutil.copyfile(src_fpath,dest_fpath)

    if True: # test set
        room_all = ['RT_0.19','RT_0.29','RT_0.39','RT_0.48','RT_0.58','RT_0.69']
        reciever_pos_all = [2,4,5,6,8,10,12,14]
        for room_i,room in enumerate(room_all):
            for reciever_pos in reciever_pos_all:
                dir = 'BRIRs/test/{}/{}'.format(room,reciever_pos)
                os.makedirs(dir)
                for azi in range(37):
                    src_fpath = 'BRIRs_GMM_backup/test/{}_{}_S{}.mat'.format(room_i,reciever_pos,azi+1)
                    dest_fpath = os.path.join(dir,'{}.mat'.format(azi))
                    shutil.copyfile(src_fpath,dest_fpath)
