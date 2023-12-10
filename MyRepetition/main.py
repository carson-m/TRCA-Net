import numpy as np
import scipy.signal as sig
from preproc import preproc
from train_trca import train_trca
from trca_recognition import trca_recognition
from itr import itr
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim

class MyDataset(Dataset): # Custom Dataset
    def __init__(self,lables,data):
        self.lables = lables # set_size,num_sample,num_character,num_subband
        self.data = data # set_size
    
    def __getitem__(self,idx):
        data_tmp = np.squeeze(self.data[idx,:,:,:])
        lable_tmp = self.lables[idx]
        return data_tmp.astype(np.float32),lable_tmp
    def __len__(self):
        return len(self.lables)

    
class NetStage1(nn.Module):
    def __init__(self,sizes,p_dropout_firststage,p_dropout_final):
        # sizes(# subband, # sample, # character=w_n)
        super(NetStage1,self).__init__()
        self.sizes=sizes
        self.conv1 = nn.Conv2d(in_channels=sizes[0],out_channels=1,kernel_size=1,padding=0) # 1 * 125 * 40
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=120,kernel_size=[1,sizes[2]],padding=0) # 120 * 125 * 1
        self.conv3 = nn.Conv2d(in_channels=120,out_channels=120,kernel_size=[2,1],stride=[2,1]) # 120 * 62 * 1
        self.conv4 = nn.Conv2d(in_channels=120,out_channels=120,kernel_size=[10,1],padding='same') # 120 * 62 * 1
        
        self.fc = nn.Linear(in_features=7440, out_features=sizes[2])
        
        self.act = F.relu
        self.softmax = nn.Softmax(dim=1)
        self.drop1st = nn.Dropout2d(p_dropout_firststage)
        self.dropfinal = nn.Dropout2d(p_dropout_final)
    
    def forward(self, x):
        x = self.drop1st(self.conv2(self.conv1(x)))
        x = self.act(self.drop1st(self.conv3(x)))
        x = self.dropfinal(self.conv4(x))
        
        x = x.view(-1,7440)
        x = self.fc(x)
        x = self.softmax(x)
        return x

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
    dropout_first_stage = 0.1 # Dropout probability of first two dropout layers at first stage
    dropout_second_stage = 0.6 # Dropout probabilities of first two dropout layers at second stage
    dropout_final = 0.95
    epochs_first_stage = 500
    
    # Preprocess
    t_sel = t_pre_stimulus + t_visual_cue
    total_delay = t_pre_stimulus + t_visual_latency
    delay_sample_points = int(np.floor(total_delay * sample_rate))
    num_sample = int(np.floor(t_visual_cue * sample_rate))
    samples = np.arange(delay_sample_points, delay_sample_points + num_sample)
    all_data, all_data_y = preproc('..\..\Data\Benchmark', channels, samples, num_character, num_block) # GET DATA all_data: preprocessed data, all_data_y: labels
    # all_data: (# channels, # sample, # characters, # blocks, # subjects)
    # all_data_y: (1, # channels, # blocks, # subjects)
    num_subject = all_data.shape[4] # number of subjects
    
    # Get bandpass filters
    bpFilters = []
    for i in range(num_subband):
        filt_tmp = sig.iirfilter(filter_order, [low_cutoff[i], high_cutoff[i]],\
            btype='bandpass', ftype='cheby1', output='sos', fs=sample_rate, rp=passband_ripple)
        bpFilters.append(filt_tmp)
    
    # Bandpass filtering
    filtered_data = np.zeros([num_channel,num_sample,num_character,num_block,num_subject,num_subband]) # Data processed by filterbank
    for i in range(num_subband):
        filtered_data[:,:,:,:,:,i] = sig.sosfiltfilt(bpFilters[i],all_data,axis=1)
    
    ground_truth = np.arange(num_character)+1
    accuracy_trca = np.zeros([num_subject,num_block])
    itr_trca = np.zeros([num_subject,num_block])
    net_train_data_tmp = np.zeros([num_subband,num_sample,num_character,num_character,num_block-1,num_subject]) #subband,#sample,#character,#w,#block,#subject
    net_test_data_tmp = np.zeros([num_subband,num_sample,num_character,num_character,num_subject]) #subband,#sample,#character,#w,#subject
    for block_i in range(num_block):
        # cross validation: leave one out
        training_blocks = np.delete(np.arange(num_block),block_i)
        trca_train_data = filtered_data[:,:,:,training_blocks,:,:] # channel,sample,character,block,subject,subband
        trca_test_data = np.squeeze(filtered_data[:,:,:,block_i,:,:]) # channel,sample,character,subject,subband
        # TRCA Analysis
        # model.w(num_character,num_subband,num_channel)
        for subject_i in range(num_subject):
            model = train_trca(np.squeeze(trca_train_data[:,:,:,:,subject_i,:]).transpose(2,0,1,3,4),sample_rate)
            result = trca_recognition(np.squeeze(trca_test_data[:,:,:,subject_i,:]).transpose(2,0,1,3),model,is_ensemble)
            is_correct = (result == ground_truth)
            accuracy_trca[subject_i,block_i] = np.mean(is_correct)
            itr_trca[subject_i,block_i] = itr(num_character,accuracy_trca[subject_i,block_i], t_sel)
            print("Subject %d, Test Block %d, Accuracy: %.2f%%, ITR: %.2f" % (subject_i, block_i, accuracy_trca[subject_i,block_i]*100,itr_trca[subject_i,block_i]))
            for character_i in range(num_character):
                for subband_i in range(num_subband):
                    for w_i in range(character_i):
                        for blk_i in range(num_block-1):
                            temp = np.squeeze(trca_train_data[:,:,character_i,blk_i,subject_i,subband_i]).transpose()
                            net_train_data_tmp[subband_i,:,character_i,w_i,blk_i,subject_i] = np.dot(temp,model['w'][w_i,subband_i,:])
                        net_test_data_tmp[subband_i,:,character_i,w_i,subject_i] = np.dot(np.squeeze(trca_test_data\
                            [:,:,character_i,subject_i,subband_i]).transpose(),model['w'][w_i,subband_i,:])
        
        train_set_size = num_character*(num_block-1)*num_subject
        net_train_data = net_train_data_tmp.transpose([2,4,5,0,1,3]).reshape([train_set_size,num_subband,num_sample,num_character])
        net_train_y = all_data_y[:,training_blocks,:].reshape([train_set_size,1]).squeeze()
        
        test_set_size = num_character*num_subject
        net_test_data = net_test_data_tmp.transpose([2,4,0,1,3]).reshape([test_set_size,num_subband,num_sample,num_character])
        net_test_y = all_data_y[:,block_i,:].squeeze().reshape([test_set_size,1]).squeeze()
        
        # Net Training Stage:1
        sizes = net_train_data.shape[1:4] # sample, # character, # subband
        net1 = NetStage1(sizes,dropout_first_stage,dropout_final) # net for stage 1
        train_set_first_stage = MyDataset(net_train_y,net_train_data)
        test_set_first_stage = MyDataset(net_test_y,net_test_data)
        train_loader_first_stage = DataLoader(train_set_first_stage,batch_size=100,shuffle=True,num_workers=8)
        test_loader_first_stage = DataLoader(test_set_first_stage,batch_size=100,shuffle=False,num_workers=8)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net1.parameters(),lr=0.0001)
        train_loss_first_stage = []
        test_loss_first_stage = []
        accuracies = []
        for epoch in range(epochs_first_stage):
            train_loss = 0.0
            test_loss = 0.0
            net1.train() # Switch to training mode
            for idx,(data,label) in tqdm(enumerate(train_loader_first_stage)):
                optimizer.zero_grad() # reset gradient to zero
                output = net1(data)
                loss = criterion(output,label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.shape[0]
            
            net1.eval() # Switch to evaluation mode
            correct = 0
            total = 0
            for idx,(data,label) in tqdm(enumerate(test_loader_first_stage)):
                output = net1(data)
                loss = criterion(output,label)
                test_loss += loss.item() * data.shape[0]
                __,predicted = torch.max(output.data,1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            train_loss = train_loss / train_set_size
            test_loss = test_loss / test_set_size
            train_loss_first_stage.append(train_loss)
            test_loss_first_stage.append(test_loss)
            accuracy = correct / total
            accuracies.append(accuracy)
            
            print(f"TestBlock:{block_i}, Epoch:{epoch}, Acc:{correct/total}, Train Loss:{train_loss}, Test Loss:{test_loss}")


if __name__ == '__main__':
    main()