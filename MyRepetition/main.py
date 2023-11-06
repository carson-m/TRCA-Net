import numpy as np
import scipy.signal as sig
from preproc import preproc

def main():
    # set parameters
    is_ensemble = True # Use Ensemble TRCA
    t_pre_stimulus = 0.5 # time before stimulus[s]
    t_visual_latency = 0.14 # time visual latency[s]
    t_visual_cue = 0.5 # time visual cue[s]
    sample_rate = 250 # sample rate[Hz]
    channels = [47, 53, 54, 55, 56, 57, 60, 61, 62] # Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2
    subband_num = 3 # number of subbands
    num_cha = 40 # number of characters
    num_blk = 6 # number of blocks
    num_cnl = len(channels) # number of channels
    filter_order = 2 # filter order
    passband_ripple = 1 # passband ripple[dB]
    high_cutoff = np.ones(subband_num) * 90 # high cutoff frequency[Hz]
    low_cutoff = np.arange(8, 8, 8 * (subband_num + 1)) # low cutoff frequency[Hz]
    
    # Preprocess
    total_delay = t_pre_stimulus + t_visual_latency
    delay_sample_points = int(total_delay * sample_rate)
    visual_cue_sample_points = int(t_visual_cue * sample_rate)
    samples = np.arange(delay_sample_points, delay_sample_points + visual_cue_sample_points)
    all_data, all_data_y = preproc('data', channels, samples, num_cha, num_blk) # GET DATA
    # all_data: (# channels, # sample length, # characters, # blocks, # of subjects)
    # all_data_y: (1, # channels, # blocks, # subjects)
    subject_num = all_data.shape[4] # number of subjects
    
    # Get bandpass filters
    bpFilters = []
    for i in range(subband_num):
        filt_tmp = sig.iirfilter(filter_order, [2*low_cutoff[i]/sample_rate, 2*high_cutoff[i]/sample_rate],\
            btype='bandpass', ftype='cheby1', output='sos', fs=sample_rate, rp=passband_ripple)
        bpFilters.append(filt_tmp)
    
    # Evaluations
    acc_mtx = np.zeros((subject_num, num_blk)) # init accuracy matrix
    acc_mtx_1 = np.zeros((subject_num, num_blk)) # init accuracy matrix for 1st stage
    acc_mtx_2 = np.zeros((subject_num, num_blk)) # init accuracy matrix for 2nd stage
    acc_trca = np.zeros((subject_num, num_blk)) # init accuracy matrix for TRCA
    
    # Cross validation
    for block in range(num_blk):
        allblock = np.arange(num_blk)
        allblock[block] = [] # Exclude the block for validation
        
        train = all_data[:, :, :, allblock, :]
        test = np.squeeze(all_data[:, :, :, block, :])
        train_tmp = np.zeros((num_cha, num_cnl, visual_cue_sample_points, num_blk))
        test_tmp = np.zeros((num_cha, num_cnl, visual_cue_sample_points))
        for subject in range(subject_num):
            train_tmp = train[:,:,:,:,subject].transpose(2, 0, 1, 3)
            