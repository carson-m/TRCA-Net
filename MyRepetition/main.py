import numpy as np
import scipy.signal as sig
import scipy.io as sio
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
import multiprocessing
import os

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

    
class DNN(nn.Module):
    def __init__(self,sizes,p_dropout_1,p_dropout_2):
        # sizes(# subband, # sample, # character=w_n)
        super(DNN,self).__init__()
        self.sizes=sizes
        self.conv1 = nn.Conv2d(in_channels=sizes[0],out_channels=1,kernel_size=1,padding=0,bias=False) # 1 * 125 * 40
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=120,kernel_size=[1,sizes[2]],padding=0) # 120 * 125 * 1
        self.conv3 = nn.Conv2d(in_channels=120,out_channels=120,kernel_size=[2,1],stride=[2,1]) # 120 * 62 * 1
        self.conv4 = nn.Conv2d(in_channels=120,out_channels=120,kernel_size=[10,1],padding='same') # 120 * 62 * 1
        
        self.fc = nn.Linear(in_features=7440, out_features=sizes[2])
        
        self.act = F.relu
        self.softmax = nn.Softmax(dim=1)
        self.drop1st = nn.Dropout2d(p_dropout_1)
        self.dropfinal = nn.Dropout2d(p_dropout_2)
    
    def forward(self, x):
        x = self.drop1st(self.conv2(self.conv1(x)))
        x = self.act(self.drop1st(self.conv3(x)))
        x = self.dropfinal(self.conv4(x))
        
        x = x.view(-1,7440)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
    # define function "transfer" to transfer the weights from the trained model to the new model
    def transfer(self, stage1_model):
        self.conv1.weight.data = stage1_model.conv1.weight.data
        
        self.conv2.weight.data = stage1_model.conv2.weight.data
        self.conv2.bias.data = stage1_model.conv2.bias.data
        
        self.conv3.weight.data = stage1_model.conv3.weight.data
        self.conv3.bias.data = stage1_model.conv3.bias.data
        
        self.conv4.weight.data = stage1_model.conv4.weight.data
        self.conv4.bias.data = stage1_model.conv4.bias.data
        
        self.fc.weight.data = stage1_model.fc.weight.data
        self.fc.bias.data = stage1_model.fc.bias.data

def train_net(device, train_loader, test_loader, train_set_size, test_set_size, sizes, dropout_1, dropout_2, num_epochs, info, transfer_net=None):
    #sizes: # sample, # character, # subband
    #train_loader: DataLoader
    #test_loader: DataLoader
    #train_set_size: size of training set
    #test_set_size: size of test set
    #num_epochs: number of epochs
    #info: information of training for printing
    #dropout_1: dropout probability of first dropout layer
    #dropout_2: dropout probability of second dropout layer
    
    net = DNN(sizes,dropout_1,dropout_2) # net for stage 1
    net = net.to(device)
    if transfer_net is not None:
        net.transfer(transfer_net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0001)
    train_loss_array = []
    test_loss_array = []
    accuracy_array = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        test_loss = 0.0
        net.train() # Switch to training mode
        for __,(data,label) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad() # reset gradient to zero
            output = net(data)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.shape[0]
                
        net.eval() # Switch to evaluation mode
        correct = 0
        total = 0
        for __,(data,label) in tqdm(enumerate(test_loader)):
            data = data.to(device)
            label = label.to(device)
            output = net(data)
            loss = criterion(output,label)
            test_loss += loss.item() * data.shape[0]
            __,predicted = torch.max(output.data,1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        train_loss = train_loss / train_set_size
        test_loss = test_loss / test_set_size
        train_loss_array.append(train_loss)
        test_loss_array.append(test_loss)
        accuracy = correct / total
        accuracy_array.append(accuracy)
            
        print(f"{info}, Epoch:{epoch}, Acc:{correct/total}, Train Loss:{train_loss}, Test Loss:{test_loss}")
        
    return net, train_loss_array, test_loss_array, accuracy_array

def main():
    USE_CUDA = True
    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    maximum_thread = multiprocessing.cpu_count()
    print('Available threads:', maximum_thread)
    num_workers = 8
    print('Num workers:', num_workers)
    
    result_folder = "test0"
    result_folder = './' + result_folder
    
    # set parameters
    is_ensemble = True # Use Ensemble TRCA or not
    transfer_learning = True # Use Transfer Learning or not
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
    epoch_transfer = 1000
    epochs_no_transfer = 2000
    parameters = {'is_ensemble':is_ensemble,'transfer_learning':transfer_learning,'t_pre_stimulus':t_pre_stimulus,\
        't_visual_latency':t_visual_latency,'t_visual_cue':t_visual_cue,'sample_rate':sample_rate,'channels':channels,\
        'num_subband':num_subband,'num_character':num_character,'num_block':num_block,'num_channel':num_channel,\
        'filter_order':filter_order,'passband_ripple':passband_ripple,'high_cutoff':high_cutoff,'low_cutoff':low_cutoff,\
        'dropout_first_stage':dropout_first_stage,'dropout_second_stage':dropout_second_stage,'dropout_final':dropout_final,\
        'epochs_first_stage':epochs_first_stage,'max_epochs_first_stage':epoch_transfer,\
        'max_epochs_second_stage':epochs_no_transfer}
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    sio.savemat(result_folder + '/parameters.mat', parameters)
    
    # Preprocess
    t_sel = t_pre_stimulus + t_visual_cue
    total_delay = t_pre_stimulus + t_visual_latency
    delay_sample_points = int(np.floor(total_delay * sample_rate))
    num_sample = int(np.floor(t_visual_cue * sample_rate))
    samples = np.arange(delay_sample_points, delay_sample_points + num_sample)
    all_data, all_data_y = preproc('../../Data/Benchmark', channels, samples, num_character, num_block) # GET DATA all_data: preprocessed data, all_data_y: labels
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
    # initialize data for net training
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
        
        if not os.path.exists(result_folder + '/testblock' + str(block_i)):
            os.makedirs(result_folder + '/testblock' + str(block_i))
        sio.savemat(result_folder + '/testblock' + str(block_i) + '/trca_result.mat', {'accuracy_trca':accuracy_trca,'itr_trca':itr_trca})
        
        train_set_size = num_character*(num_block-1)*num_subject
        net_train_data = net_train_data_tmp.transpose([2,4,5,0,1,3]).reshape([train_set_size,num_subband,num_sample,num_character])
        net_train_y = all_data_y[:,training_blocks,:].reshape([train_set_size,1]).squeeze()
        
        test_set_size = num_character*num_subject
        net_test_data = net_test_data_tmp.transpose([2,4,0,1,3]).reshape([test_set_size,num_subband,num_sample,num_character])
        net_test_y = all_data_y[:,block_i,:].squeeze().reshape([test_set_size,1]).squeeze()
        
        # Net Training Stage:1
        sizes = net_train_data.shape[1:4] # sample, # character, # subband
        net1 = DNN(sizes,dropout_first_stage,dropout_final) # net for stage 1
        net1 = net1.to(device)
        train_set_first_stage = MyDataset(net_train_y,net_train_data)
        test_set_first_stage = MyDataset(net_test_y,net_test_data)
        train_loader_first_stage = DataLoader(train_set_first_stage,batch_size=100,shuffle=True,num_workers = num_workers)
        test_loader_first_stage = DataLoader(test_set_first_stage,batch_size=100,shuffle=False,num_workers = num_workers)
        
        net1, train_loss_first_stage, test_loss_first_stage, accuracies = train_net\
            (device, train_loader_first_stage, test_loader_first_stage, train_set_size,\
                test_set_size, sizes, dropout_first_stage, dropout_final, epochs_first_stage, "1st Stage, TestBlock:" + str(block_i))
        
        sio.savemat(result_folder + '/testblock' + str(block_i) + '/net1_result.mat', {'train_loss':train_loss_first_stage,'test_loss':test_loss_first_stage,'accuracy':accuracies})
        
        # Net Training Stage:2
        if transfer_learning:
            epochs_stage2 = epoch_transfer
        else:
            epochs_stage2 = epochs_no_transfer
        for s in range(num_subject):
            # Obtain subject specific data
            train_set_size_subject = num_character*(num_block-1)
            net_train_data_subject = np.squeeze(net_train_data_tmp[:,:,:,:,:,s]).transpose([2,4,0,1,3]).\
                reshape([train_set_size_subject,num_subband,num_sample,num_character])
            all_data_y_subject = all_data_y[:,training_blocks,:]
            net_train_y_subject = (all_data_y_subject[:,:,s].squeeze()).reshape([train_set_size_subject,1]).squeeze()
        
            test_set_size_subject = num_character
            net_test_data_subject = net_test_data_tmp[:,:,:,:,s].squeeze()\
                .transpose([2,0,1,3]).reshape([test_set_size_subject,num_subband,num_sample,num_character])
            all_data_y_subject = all_data_y[:,block_i,:].squeeze()
            net_test_y_subject = all_data_y_subject[:,s].squeeze()
            train_set_second_stage = MyDataset(net_train_y_subject,net_train_data_subject)
            test_set_second_stage = MyDataset(net_test_y_subject,net_test_data_subject)
            train_loader_second_stage = DataLoader(train_set_second_stage,batch_size=num_character*(num_block-1),shuffle=True,num_workers=num_workers)
            test_loader_second_stage = DataLoader(test_set_second_stage,batch_size=num_character,shuffle=False,num_workers=num_workers)
            sizes = net_train_data.shape[1:4] # sample, # character, # subband
            __, train_loss_second_stage, test_loss_second_stage, accuracies_second_stage = train_net\
                (device, train_loader_second_stage, test_loader_second_stage, train_set_size_subject,\
                    test_set_size_subject, sizes, dropout_second_stage, dropout_final, epochs_stage2,\
                        "2nd Stage, TestBlock:" + str(block_i) + ", Subject:" + str(s), net1)
        
            sio.savemat(result_folder + '/testblock' + str(block_i) + '/net2_result_Subject_' + str(s) + '_.mat', \
                {'train_loss':train_loss_second_stage,'test_loss':test_loss_second_stage,'accuracy':accuracies_second_stage})

if __name__ == '__main__':
    main()