import numpy as np

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
    
def train_trca(eeg_dat,fs):
    # eeg_dat (character,channel,sample,block,subband)
    num_target,num_channel,num_sample,__,num_subband = eeg_dat.shape
    # traindata = np.zeros((num_target,num_subband,num_channel,num_sample))
    traindata = np.mean(eeg_dat,axis=3).transpose(0,3,1,2)
    w = np.zeros((num_target,num_subband,num_channel))
    for targ_i in range(num_target):
        eeg_tmp = np.squeeze(eeg_dat[targ_i,:,:,:,:])
        for subband_i in range(num_subband):
            w[targ_i,subband_i,:] = trca(np.squeeze(eeg_tmp[:,:,:,subband_i]))[:,0]
    return {'traindata':traindata,'w':w,'fs':fs,'num_targ':num_target,'num_filt':num_subband}