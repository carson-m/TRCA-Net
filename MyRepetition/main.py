import numpy as np
import scipy.signal as sig
from preproc import preproc
from train_trca import train_trca
from trca_recognition import trca_recognition
from itr import itr

def main():
    # set parameters
    is_ensemble = False # Use Ensemble TRCA or not
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
    low_cutoff = np.arange(8, 8 * (subband_num + 1), 8) # low cutoff frequency[Hz]
    
    # Preprocess
    t_sel = t_pre_stimulus + t_visual_cue
    total_delay = t_pre_stimulus + t_visual_latency
    delay_sample_points = int(total_delay * sample_rate)
    visual_cue_sample_points = int(t_visual_cue * sample_rate)
    samples = np.arange(delay_sample_points, delay_sample_points + visual_cue_sample_points)
    all_data, all_data_y = preproc('..\..\Data\Benchmark', channels, samples, num_cha, num_blk) # GET DATA all_data: preprocessed data, all_data_y: labels
    # all_data: (# channels, # sample length, # characters, # blocks, # of subjects)
    # all_data_y: (1, # channels, # blocks, # subjects)
    subject_num = all_data.shape[4] # number of subjects
    
    # Get bandpass filters
    bpFilters = []
    for i in range(subband_num):
        filt_tmp = sig.iirfilter(filter_order, [2*low_cutoff[i]/sample_rate, 2*high_cutoff[i]/sample_rate],\
            btype='bandpass', ftype='cheby1', output='sos', fs=sample_rate, rp=passband_ripple)
        bpFilters.append(filt_tmp)
    
    # Evaluation matrices
    acc_mtx = np.zeros([subject_num, num_blk]) # init accuracy matrix
    acc_mtx_1 = np.zeros([subject_num, num_blk]) # init accuracy matrix for 1st stage
    acc_mtx_2 = np.zeros([subject_num, num_blk]) # init accuracy matrix for 2nd stage
    acc_trca = np.zeros([subject_num, num_blk]) # init accuracy matrix for TRCA
    itr_trca = np.zeros([subject_num, num_blk]) # init itr matrix for TRCA
    reference_result = range(0, num_cha) # reference result
    
    # Auxillary Matrices
    train_tmp = np.zeros([num_cha, num_cnl, visual_cue_sample_points, num_blk])
    test_tmp = np.zeros([num_cha, num_cnl, visual_cue_sample_points])
    filtered_train_tmp = np.zeros([subband_num, num_cha, num_cnl, visual_cue_sample_points, num_blk])
    filtered_test_tmp = np.zeros([subband_num, num_cha, num_cnl, visual_cue_sample_points])
    
    # Cross validation
    for block in range(num_blk):
        allblock = np.arange(num_blk)
        allblock[block] = [] # Exclude the block for validation
        
        train = all_data[:, :, :, allblock, :]
        test = np.squeeze(all_data[:, :, :, block, :])
        for subject in range(subject_num):
            train_tmp = np.squeeze(train[:,:,:,:,subject]).transpose(2, 0, 1, 3)
            model = train_trca(train_tmp, sample_rate. subband_num)
            test_tmp = np.squeeze(test[:,:,:,subject]).transpose(2, 0, 1)
            result = trca_recognition(test_tmp, model, is_ensemble)
            correct = result == reference_result
            acc_trca[subject, block] = np.mean(correct) * 100
            itr_trca[subject, block] = itr(num_cha, acc_trca[subject, block] / 100, t_sel)
            for subband in range(subband_num):
                filtered_train_tmp[subband, :, :, :, :] = sig.sosfiltfilt(bpFilters[subband], train_tmp, axis=2)
                filtered_test_tmp[subband, :, :, :, :] = sig.sosfiltfilt(bpFilters[subband], test_tmp, axis=2)

if __name__ == '__main__':
    main()