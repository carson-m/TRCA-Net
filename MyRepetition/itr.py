import numpy as np

def itr(n,p,t):
    # n: number of targets
    # p: accuracy
    # t: avg time for a selection [s]
    # returns info transfer rate [bits/min]
    
    if p<0 or p>1:
        raise ValueError('p must be between 0 and 1')
    elif p < 1/n:
        print('p must be greater than chance level')
    elif p == 1:
        return np.log2(n)*60/t
    else:
        return (np.log2(n)+p*np.log2(p)+(1-p)*np.log2((1-p)/(n-1)))*60/t
        