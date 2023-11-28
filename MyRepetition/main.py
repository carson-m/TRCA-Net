import numpy as np
import scipy.signal as sig
from preproc import preproc
from train_trca import train_trca
from trca_recognition import trca_recognition
from itr import itr

def main():
    # set parameters
    is_ensemble = True # Use Ensemble TRCA or not
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
    filtered_data = np.zeros([num_channel,num_sample,num_character,num_block,num_subject,num_subband]) # Data processed by filterbank
    for i in range(num_subband):
        filtered_data[:,:,:,:,:,i] = sig.sosfiltfilt(bpFilters[i],all_data,axis=1)
    
    ground_truth = np.arange(num_character)
    accuracy_trca = np.zeros([num_subject,num_block])
    net_train_data_tmp = np.zeros([num_subband,num_sample,num_character,num_character,num_block-1,num_subject]) #subband,#sample,#character,#w,#block,#subject
    net_test_data_tmp = np.zeros([num_subband,num_sample,num_character,num_character,num_subject]) #subband,#sample,#character,#w,#subject
    for block_i in range(num_block):
        # cross validation: leave one out
        training_blocks = range(num_block)
        training_blocks[block_i]=[]
        trca_train_data = filtered_data[:,:,:,training_blocks,:,:] # channel,sample,character,block,subject,subband
        trca_test_data = np.squeeze(filtered_data[:,:,:,block_i,:,:]) # channel,sample,character,subject,subband
        # TRCA Analysis
        # model.w(num_character,num_subband,num_channel)
        for subject_i in range(num_subject):
            model = train_trca(np.squeeze(trca_train_data[:,:,:,:,subject_i,:]).transpose(2,0,1,3,4),sample_rate)
            result = trca_recognition(np.squeeze(trca_test_data[:,:,:,subject_i,:]).transpose(2,0,1,3))
            is_correct = (result == ground_truth)
            accuracy_trca[subject_i,block_i] = np.mean(is_correct)
            print("Subject %d, Test Block %d, Accuracy: %.2f%%" % (subject_i, block_i, accuracy_trca[subject_i,block_i]*100))
            for character_i in range(num_character):
                for subband_i in range(num_subband):
                    for w_i in range(character_i):
                        for blk_i in range(num_block-1):
                            net_train_data_tmp[subband_i,:,character_i,w_i,blk_i,subject_i] = np.squeeze(trca_train_data\
                                [:,:,character_i,blk_i,subject_i,subband_i]).transpose()*model['w'][subband_i,w_i,:]
                        net_test_data_tmp[subband_i,:,character_i,w_i,subject_i] = np.squeeze(trca_test_data\
                            [:,:,character_i,subject_i,subband_i]).transpose()*model['w'][subband_i,w_i]
        
        train_set_size = num_character*(num_block-1)*num_subject
        net_train_data = net_train_data_tmp.transpose([2,4,5,1,3,0]).reshape([train_set_size,num_sample,num_character,num_subband])
        net_train_y = all_data_y[:,training_blocks,:].reshape([train_set_size,1]).squeeze()
        
        test_set_size = num_character*num_subject
        net_test_data = net_test_data_tmp.transpose([2,4,1,3,0]).reshape([test_set_size,num_sample,num_character,num_subband])
        net_test_y = all_data_y[:,block_i,:].squeeze().reshape([test_set_size,1]).squeeze()
        


if __name__ == '__main__':
    main()