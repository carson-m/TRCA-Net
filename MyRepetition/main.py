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
    num_subband = 3 # number of subbands
    num_character = 40 # number of characters
    num_block = 6 # number of blocks
    num_channel = len(channels) # number of channels
    filter_order = 2 # filter order
    passband_ripple = 1 # passband ripple[dB]
    high_cutoff = np.ones(num_subband) * 90 # high cutoff frequency[Hz]
    low_cutoff = np.arange(8, 8 * (num_subband + 1), 8) # low cutoff frequency[Hz]
    
    # Preprocess
    t_sel = t_pre_stimulus + t_visual_cue
    total_delay = t_pre_stimulus + t_visual_latency
    delay_sample_points = int(total_delay * sample_rate)
    num_sample = int(t_visual_cue * sample_rate)
    samples = np.arange(delay_sample_points, delay_sample_points + num_sample)
    all_data, all_data_y = preproc('..\..\Data\Benchmark', channels, samples, num_character, num_block) # GET DATA all_data: preprocessed data, all_data_y: labels
    # all_data: (# channels, # sample, # characters, # blocks, # subjects)
    # all_data_y: (1, # channels, # blocks, # subjects)
    num_subject = all_data.shape[4] # number of subjects
    
    # Get bandpass filters
    bpFilters = []
    for i in range(num_subband):
        filt_tmp = sig.iirfilter(filter_order, [2*low_cutoff[i]/sample_rate, 2*high_cutoff[i]/sample_rate],\
            btype='bandpass', ftype='cheby1', output='sos', fs=sample_rate, rp=passband_ripple)
        bpFilters.append(filt_tmp)
    
    # Bandpass filtering
    filtered_data = np.zeros([num_channel,num_sample,num_character,num_block,num_subject,num_subband])
    for i in range(num_subband):
        filtered_data[:,:,:,:,:,i] = sig.sosfiltfilt(bpFilters[i],all_data,axis=1)
    
    for block_i in range(num_block):
        # cross validation: leave one out
        block_selection = range(num_block)
        block_selection[block_i]=[]
        train_data = all_data[:,:,:,block_selection,:]
        test_data = np.squeeze(all_data[:,:,:,block_i,:])
        # TRCA Analysis
        W = np.zeros([num_subject,num_character,num_subband,num_channel])
        for subject_i in range(num_subject):
            model = train_trca(np.squeeze(train_data[subject_i,:,:,:,:]),sample_rate)
            W[subject_i,:,:,:] = model['w']
            
    
        
    # # Auxillary Matrices
    # train_tmp = np.zeros([num_character, num_channel, visual_cue_sample_points, num_block])
    # test_tmp = np.zeros([num_character, num_channel, visual_cue_sample_points])
    # filtered_train_tmp = np.zeros([num_subband, num_character, num_channel, visual_cue_sample_points, num_block])
    # filtered_test_tmp = np.zeros([num_subband, num_character, num_channel, visual_cue_sample_points])
    
    # # Cross validation
    # for block in range(num_block):
    #     allblock = np.arange(num_block)
    #     allblock[block] = [] # Exclude the block for validation
        
    #     train = all_data[:, :, :, allblock, :]
    #     test = np.squeeze(all_data[:, :, :, block, :])
    #     for subject in range(subject_num):
    #         train_tmp = np.squeeze(train[:,:,:,:,subject]).transpose(2, 0, 1, 3)
    #         model = train_trca(train_tmp, sample_rate. num_subband)
    #         test_tmp = np.squeeze(test[:,:,:,subject]).transpose(2, 0, 1)
    #         result = trca_recognition(test_tmp, model, is_ensemble)
    #         correct = result == reference_result
    #         acc_trca[subject, block] = np.mean(correct) * 100
    #         itr_trca[subject, block] = itr(num_character, acc_trca[subject, block] / 100, t_sel)
    #         for subband in range(num_subband):
    #             filtered_train_tmp[subband, :, :, :, :] = sig.sosfiltfilt(bpFilters[subband], train_tmp, axis=2)
    #             filtered_test_tmp[subband, :, :, :, :] = sig.sosfiltfilt(bpFilters[subband], test_tmp, axis=2)

if __name__ == '__main__':
    main()