import numpy as np
import os
import RT60

def cal_test_A_room():
    room_size = (5.1,7.1,3)
    freq_all = np.asarray([125,250,500,1000,2000,4000])
    RT_room_all = np.asarray([[0.48,0.33,0.13,0.08,0.06,0.06],
                              [0.67,0.48,0.24,0.15,0.11,0.10],
                              [0.81,0.61,0.36,0.23,0.18,0.15],
                              [0.93,0.75,0.46,0.30,0.24,0.19],
                              [1.09,0.89,0.56,0.39,0.30,0.24],
                              [1.26,1.03,0.69,0.48,0.37,0.29]])
    n_room,n_band = RT_room_all.shape

    A_room_all = np.zeros((n_room,n_band))
    for room_i in range(n_room):
        A_room_all[room_i] = RT60.RT2coef(RT60=RT_room_all[room_i],
                                          freq_all=freq_all,room_size=room_size)

    return A_room_all


def gen_textsetup():

    roomsim_fdir = '/home/st/Disks/Disk1/Work_Space/my_module/Roomsim_Campbell_st'
    temple_fpath = os.path.join(roomsim_fdir,'Text_setups_GMM/template.txt')
    with open(temple_fpath,'r') as setting_temp_file:
        setting_str_temp = setting_temp_file.read()

    A_room_all = cal_test_A_room()
    pos_label_all = [2,4,5,6,8,10,12,14]

    for room_i in range(6):
        for pos_label in pos_label_all:
            pos_x = 0.5+np.mod(pos_label-1,5)*0.5
            pos_y = 3.50-((pos_label-1)/5)
            pos_z = 1.75
            A_wall_str = ' '.join(['{:.4f}'.format(coef) for coef in A_room_all[room_i]])
            setting_str = setting_str_temp.format(
                                recieve_pos_x=pos_x,recieve_pos_y=pos_y,
                                recieve_pos_z=pos_z,
                                fname='{}_{}'.format(room_i,pos_label),
                                A_x1=A_wall_str,A_x2=A_wall_str,
                                A_y1=A_wall_str,A_y2=A_wall_str,
                                A_z1=A_wall_str,A_z2=A_wall_str)
            setup_fpath = os.path.join(roomsim_fdir,
                       'Text_setups_GMM','{}_{}.txt'.format(room_i,pos_label))

            with open(setup_fpath,'w') as setup_file:
                setup_file.write(setting_str)

if __name__ == '__main__':
    gen_textsetup()
