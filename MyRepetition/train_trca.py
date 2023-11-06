import numpy as np
from filterbank import filterbank

def trca(eeg_dat):
    # eeg_dat: num_chans x num_samples(data length) x num_trials
    num_chans, num_samples, num_trials = eeg_dat.shape
    S = np.zeros(num_chans)
    for trial_i in range(num_trials-1):
        x1 = np.squeeze(eeg_dat[:,:,trial_i])
        x1 = x1-np.mean(x1, axis=1, keepdims=True)
        for trial_j in range(trial_i+1, num_trials):
            x2 = np.squeeze(eeg_dat[:,:,trial_j])
            x2 = x2-np.mean(x2, axis=1, keepdims=True)
            S = S + np.dot(x1, x2.T) + np.dot(x2, x1.T)
    UX = np.reshape(eeg_dat, (num_chans, num_samples*num_trials))
    UX = UX-np.mean(UX, axis=1, keepdims=True)
    Q = np.dot(UX, UX.T)
    __,W = np.linalg.eig(np.dot(np.linalg.inv(Q), S))
    return W
    
def train_trca(eeg_dat,fs,num_filt):
    num_targets,num_channels,num_samples,__ = eeg_dat.shape
    traindata = np.zeros((num_targets,num_filt,num_channels,num_samples))
    w = np.zeros((num_targets,num_filt,num_channels))
    for targ_i in range(num_targets):
        eeg_tmp = np.squeeze(eeg_dat[targ_i,:,:,:])
        for fb_i in range(num_filt):
            eeg_tmp = filterbank(eeg_tmp,fs,fb_i)
            traindata[targ_i,fb_i,:,:] = np.squeeze(np.mean(eeg_tmp, axis=2))
            # w_temp = trca(eeg_tmp)[:,0]
            # max_idx = np.argmax(np.abs(w_temp))
            # factor = 1/w_temp[max_idx]
            # w[targ_i,fb_i,:] = w_temp*factor
            w[targ_i,fb_i,:] = trca(eeg_tmp)[:,0]
    return {'traindata':traindata,'w':w,'fs':fs,'num_targ':num_targets,'num_filt':num_filt}