import os
import shutil

# convert azi label to degree

# model
model_dir_orig = 'models/all_room'
model_dir_new = 'models1/all_room'
# os.makedirs(model_dir_new)
for azi_label in range(8,29):
    for band_i in range(32):
        azi = azi_label*5-90
        src_fpath = os.path.join(model_dir_orig,'{}_{}.npy'.format(azi_label,band_i))
        dest_fpath = os.path.join(model_dir_new,'{}_{}.npy'.format(azi,band_i))
        shutil.copyfile(src=src_fpath,dst=dest_fpath)
        print(src_fpath)
