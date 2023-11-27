import numpy as np

def trca_recognition(eeg_dat,model,is_ensemble):
    # eeg_dat(# target, # channel, # sample, # subband)
    fb_coeffs = np.power(np.array(range(model['num_filt']))+1,-1.25)+0.25
    result = np.zeros(model['num_targ'])
    for target_idx in range(model['num_targ']):
        eeg_tmp = np.squeeze(eeg_dat[target_idx,:,:,:])
        r_square = np.zeros((model['num_filt'],model['num_targ']))
        for fb_idx in range(model['num_filt']):
            eeg_filtered = np.squeeze(eeg_tmp[:,:,fb_idx])
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