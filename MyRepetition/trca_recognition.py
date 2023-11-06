from filterbank import filterbank
import numpy as np
from filterbank import filterbank
from itr import itr
from train_trca import train_trca
import scipy as sp

def trca_recognition(eeg_dat,model,is_ensemble):
    fs = model['fs']
    fb_coeffs = np.power(np.array(range(model['num_filt']))+1,-1.25)+0.25
    result = np.zeros(model['num_targ'])
    for target_idx in range(model['num_targ']):
        eeg_tmp = np.squeeze(eeg_dat[target_idx,:,:])
        r_square = np.zeros((model['num_filt'],model['num_targ']))
        for fb_idx in range(model['num_filt']):
            eeg_filtered = filterbank(eeg_tmp,fs,fb_idx)
            for class_idx in range(model['num_targ']):
                train_data = model['traindata'][class_idx,fb_idx,:,:]
                if is_ensemble:
                    w = np.squeeze(model['w'][target_idx,:,:]).transpose()
                else:
                    w = np.squeeze(model['w'][target_idx,fb_idx,:])
                eeg_vec = np.dot(np.transpose(eeg_filtered),w)
                traindat_vec = np.dot(np.transpose(train_data),w)
                r_temp = np.corrcoef(eeg_vec.flatten('F'),traindat_vec.flatten('F'))
                r_square[fb_idx,class_idx] = r_temp[0,1]
        rho = np.dot(fb_coeffs,r_square)
        result[target_idx] = np.argmax(rho)+1
    return result

def my_round(val_in):
    integer = np.floor(np.abs(val_in))
    decimal = np.abs(val_in)-integer
    if decimal>=0.5:
        integer += 1
    if val_in<0:
        integer = -integer
    return int(integer)

def __main__():
    fs = 250
    len_gaze_s = 0.5 # data length for target identification [s]
    len_shift_s = 0.5 # duration for gaze shifting [s]
    len_delay_s = 0.13 # visual latency [s]
    num_targs = 40
    num_filt = 5
    is_ensemble = True
    alpha_ci = 0.05
    
    len_gaze_sample = my_round(len_gaze_s*fs)
    len_delay_sample = my_round(len_delay_s*fs)
    segment_data = np.arange(len_delay_sample,len_delay_sample+len_gaze_sample).astype(int)
    
    # data preparation
    eeg = sp.io.loadmat('sample.mat')['eeg']
    eeg = eeg[:,:,segment_data,:]
    
    len_sel_s = len_gaze_s + len_shift_s # duration for gaze selection [s]
    lables = np.arange(1,num_targs+1,1)
    target_num,__,__,trial_num = eeg.shape
    print('Is Ensemble:',is_ensemble)
    accs = np.zeros(trial_num)
    itrs = np.zeros(trial_num)
    for i in range(trial_num): # leave one out
        train_dat = eeg
        train_dat = train_dat[:,:,:,np.arange(trial_num)!=i]
        model = train_trca(train_dat,fs,num_filt)
        result = trca_recognition(eeg[:,:,:,i],model,is_ensemble)
        is_correct = result==lables
        accs[i] = np.mean(is_correct)*100
        itrs[i] = itr(target_num,accs[i]/100,len_sel_s)
        print('Trial %d: Accuracy = %.4f%%, ITR = %.4f bits/min' % (i+1,accs[i],itrs[i]))
    print('Summary:')
    mean_acc,sigma_acc = sp.stats.norm.fit(accs)
    confidence_intv = sp.stats.norm.interval(1-alpha_ci,loc=mean_acc,scale=sigma_acc)
    print('Mean accuracy = %.4f%% (95%% CI: %.4f%% - %.4f%%)' % (mean_acc,confidence_intv[0],confidence_intv[1]))
    mean_itr,sigma_itr = sp.stats.norm.fit(itrs)
    confidence_intv = sp.stats.norm.interval(0.95,loc=mean_itr,scale=sigma_itr)
    print('Mean ITR = %.4f bits/min (95%% CI: %.4f - %.4f bits/min)' % (mean_itr,confidence_intv[0],confidence_intv[1]))
    
    
if __name__ == '__main__':
    __main__()