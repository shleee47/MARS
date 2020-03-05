from dataset.dataset import *
from torch.utils.data import Dataset, DataLoader
import getpass
import os
import socket
import numpy as np
from dataset.preprocess_data import *
from PIL import Image, ImageFilter
import argparse
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from models.model import generate_model
from opts import parse_opts
from torch.autograd import Variable
import time
import torch.utils
import sys
from utils import *
import pdb
    
if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # print configuration options
    opt = parse_opts()
    print(opt)
    
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    print("Preprocessing validation data ...")
    print("Data: {} / split: {} --------------------".format(opt.dataset, opt.split))
    data = globals()['{}_test'.format(opt.dataset)](split = opt.split, train = 0, opt = opt)
    print("Length of validation data = ", len(data))
    
    if opt.modality=='RGB': opt.input_channels = 3
    elif opt.modality=='Flow': opt.input_channels = 2

    print("Preparing datatloaders ...")
    val_dataloader = DataLoader(data, batch_size = 1, shuffle=False, num_workers = opt.n_workers, pin_memory = True, drop_last=False)
    print("Length of validation datatloader = ",len(val_dataloader))
    
    # Loading model and checkpoint
    model, parameters = generate_model(opt)
    if opt.resume_path1:
        print('loading checkpoint {}'.format(opt.resume_path1))
        checkpoint = torch.load(opt.resume_path1, map_location=load_device)
        assert opt.arch == checkpoint['arch']
        model.load_state_dict(checkpoint['state_dict'])
    print('start evaluation')
    model.eval()

    accuracies = AverageMeter()
    clip_accuracies = AverageMeter()
    
    #Path to store results
    result_path = "{}/{}/".format(opt.result_path, opt.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)    

    if opt.log:
        f = open(os.path.join(result_path, "test_{}{}_{}_{}_{}_{}.txt".format( opt.model, opt.model_depth, opt.dataset, opt.split, opt.modality, opt.sample_duration)), 'w+')
        f.write(str(opt))
        f.write('\n')
        f.flush()
        
    cnt = 0
    with torch.no_grad():   
        for i, (clip, label) in enumerate(val_dataloader):
            clip = torch.squeeze(clip)
            if opt.modality == 'RGB':
                inputs = torch.Tensor(int(clip.shape[1]/opt.sample_duration), 3, opt.sample_duration, opt.sample_size, opt.sample_size)
            elif opt.modality == 'Flow':
                inputs = torch.Tensor(int(clip.shape[1]/opt.sample_duration), 2, opt.sample_duration, opt.sample_size, opt.sample_size)
                    
            for k in range(inputs.shape[0]):
                inputs[k,:,:,:,:] = clip[:,k*opt.sample_duration:(k+1)*opt.sample_duration,:,:]   

            inputs_var = Variable(inputs)
            outputs_var= model(inputs_var)

            pred5 = np.array(torch.mean(outputs_var, dim=0, keepdim=True).topk(5, 1, True)[1].cpu().data[0])
               
            acc = float(pred5[0] == label[0])
                            
            accuracies.update(acc, 1)            
            
#            line = "Video[" + str(i) + "] : \t top5 " + str(pred5) + "\t top1 = " + str(pred5[0]) +  "\t true = " +str(int(label[0])) + "\t video = " + str(accuracies.avg)
            
            ### Detect Fight with Related Classes
            fight_related_classes = {'43': 'capoeira', '105': 'drop kicking', '122': 'face planting', '150': 'headbutting', '152': 'high kick', '235': 'playing ice hockey', '258': 'punching bag', '259': 'punching person (boxing)', '302': 'side kick', '314': 'slapping', '345': 'sword fighting', '346': 'tai chi', '395': 'wrestling'}
            FightTF = False
            for pred in pred5:
                if str(pred) in fight_related_classes.keys():
                    FightTF = True

            answer = False
            if str(int(label[0])) in fight_related_classes.keys():
                answer = True

            if FightTF == answer:
                cnt +=1
            line = "Video[" + str(i) + "] : \t top5 " + str(pred5) + "\t Is Fight? = " + str(FightTF) +  "\t Answer = " + str(answer) + "\t video = " + str(accuracies.avg) + "\t // Correct: " + str(cnt)
            print(line)

            if opt.log:
                f.write(line + '\n')
                f.flush()
    
    print("Video accuracy = ", accuracies.avg)
    line = "Video accuracy = " + str(accuracies.avg) + '\n'
    if opt.log:
        f.write(line)
    
