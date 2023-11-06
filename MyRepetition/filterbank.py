from scipy import signal
import numpy as np

def filterbank(eeg_data,fs,filt_idx):
    fs = fs/2
    
    passband = [6, 14, 22]
    stopband = [4, 10, 16]
    Wp = [passband[filt_idx]/fs, 90/fs]
    Ws = [stopband[filt_idx]/fs, 100/fs]
    N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)
    B, A = signal.cheby1(N, 0.5, Wn,'bandpass')
    
    y = signal.filtfilt(B, A, eeg_data, axis=1)
    return y