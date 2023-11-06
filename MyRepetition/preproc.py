import os
import scipy.io as sio
import numpy as np
def preproc(dir, channels, samples, num_cha, num_blk):
    # parameters
    # dir: the directory of the data
    # channels: the channels that we want to use
    # samples: sample points that we want to use
    # num_cha: number of characters
    # num_blk: number of blocks
    
    # outputs
    # all_data: (# of channels, # sample length, # of characters, # of blocks, # of subjects)
    # all_data_y: labels of characters in all_data
    
    list = os.listdir(dir)
    
    # remove the files that are not .mat files
    for item in list:
        if item.endswith('.mat'):
            continue
        else:
            list.remove(item)
    
    total_subjects = len(list)        
    # concatenate the path and the file name
    for i in range(total_subjects):
        list[i] = os.path.join(dir, list[i])
    
    all_data = np.zeros(len(channels), len(samples), num_cha, num_blk, total_subjects)
    all_data_y = np.zeros(1, num_cha, num_blk, total_subjects)
    # load the data
    for  subject in range(total_subjects):
        mat_contents = sio.loadmat(list[subject])
        mat_dat = mat_contents.data[channels, samples, :, :]
        all_data[:, :, :, :, subject] = mat_dat
    
    for cha in range(num_cha):
        all_data_y[0, cha, :, :] = cha
    
    return all_data, all_data_y