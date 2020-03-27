import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
import time
import os

from fastspeech import FastSpeech
from text import text_to_sequence
from tqdm import tqdm
import hparams as hp
import utils
import Audio
import glow
import waveglow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'

def get_FastSpeech(num):
    num=50000
    checkpoint_path = "checkpoint_%08d.pth"%num
    model = FastSpeech().to(device)
    model.load_state_dict(torch.load(os.path.join(
        hp.checkpoint_path, checkpoint_path),map_location=torch.device(device))['model'])
    model.eval()

    return model


def synthesis(model, condition1, condition2,mel_in, D, alpha=1.0):
    condition1 = np.stack([condition1])
    condition2 = np.stack([condition2])
    D=np.stack([D])

    src_pos = np.array([i+1 for i in range(condition1.shape[1])])
    src_pos = np.stack([src_pos])
    with torch.no_grad():
        sequence1 = torch.autograd.Variable(
            torch.from_numpy(condition1)).to(device).long()
        sequence2 = torch.autograd.Variable(
            torch.from_numpy(condition2)).to(device).long()
        mel_in = torch.autograd.Variable(
            torch.from_numpy(mel_in)).to(device).float()
        src_pos = torch.autograd.Variable(
            torch.from_numpy(src_pos)).to(device).long()
        D = torch.autograd.Variable(
            torch.from_numpy(D)).to(device).long()

        mel= model.forward(sequence1, sequence2, mel_in,src_pos, alpha=alpha,length_target=D)

        return mel


if __name__ == "__main__":
    
    # Test
    checkpoint_in=open(os.path.join(hp.checkpoint_path, 'checkpoint.txt'),'r')
    num=int(checkpoint_in.readline().strip())
    checkpoint_in.close()
    alpha = 1.0
    model = get_FastSpeech(num)
    # condition1=np.load("/data/con1s/001300.npy")
    # condition2=np.load("/data/con2s/001300.npy")
    # D=np.load("/data/alignments/001300.npy")
    
    n=len(os.listdir('./tmp/con1s'))
    
    os.system('rm ./tmp/f0s/*')
    for i in tqdm(range(n)):
        condition1=np.load("./tmp/con1s/%03d.npy"%i)
        condition2=np.load("./tmp/con2s/%03d.npy"%i)
        D=np.load("./tmp/Ds/%03d.npy"%i)
        mel_in=np.load("./tmp/mels/%03d.npy"%i)
        
#         print(D)

        mel_output= synthesis(
            model, condition1, condition2,mel_in, D, alpha=alpha)

        predict=mel_output.transpose(1,2)
        p=F.softmax(predict,dim=1).transpose(1,2)[0].detach().cpu().numpy()
        sample=np.argmax(p,axis=1)
#         sample=[]
#         for index in range(p.shape[0]):
#             sample.append(np.random.choice(200,1,p=p[index]))
#         sample=np.array(sample)[:,0]
#         print(sample.shape)
        
        np.save('./tmp/f0s/%03d.npy'%i,sample)
        
#         f=plt.figure()
#         plt.matshow(mel_output[0].cpu().detach().numpy())
#         plt.savefig('out_predicted.png')
#         plt.matshow(F.softmax(predict,dim=1).transpose(1,2)[0].cpu().detach().numpy())
#         plt.savefig('out_predicted_softmax.png')
#         plt.cla() 

#         f=plt.figure(figsize=(8,6))
#         #                   plt.matshow(mel_target[0].cpu().detach().numpy())
#         #                   x=np.arange(mel_target.shape[1])
#         #                   y=sample_from_discretized_mix_logistic(mel_output.transpose(1,2)).cpu().detach().numpy()[0]
#         #                   plt.plot(x,y)
#         
        

        
#         smooth=[]
#         k=5
#         for index in range(sample.shape[0]-k):
#             if sample[index]>20:
#                 smooth.append(sample[index:index+k].mean())
#             else:
#                 smooth.append(0.0)
#         smooth=np.array(smooth)
#         plt.plot(np.arange(smooth.shape[0]),smooth,color='black',linewidth = '1')  

#         plt.plot(np.arange(sample.shape[0]),sample,color='grey',linewidth = '1', linestyle=':')
#         for index in range(D.shape[0]):
#             x=np.arange(D[index])+D[:index].sum()
#             y=np.arange(D[index])
#             if condition2[index]!=0:
#                 y.fill((condition2[index]-40.0)*5)
#                 plt.plot(x,y)
#         plt.savefig('out_target.png',dpi=300)
#         plt.cla() 

#         plt.close("all")


