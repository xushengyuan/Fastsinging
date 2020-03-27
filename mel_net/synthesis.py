import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import cv2

from fastspeech import FastSpeech
from text import text_to_sequence
from tqdm import tqdm
import hparams as hp
import utils
import Audio
import glow
import waveglow
import soundfile as sf
import pyworld as pw

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=''

def get_FastSpeech(num):
    checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_01400000.pth'))
    model = nn.DataParallel(FastSpeech())
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def synthesis(model, condition1, condition2, D, alpha=1.0):
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
        src_pos = torch.autograd.Variable(
            torch.from_numpy(src_pos)).to(device).long()
        D = torch.autograd.Variable(
            torch.from_numpy(D)).to(device).long()

        mel, mel_postnet = model.forward(sequence1, sequence2, src_pos, alpha=alpha,length_target=D)

        return mel[0], \
            mel_postnet[0], \
            mel.transpose(1, 2), \
            mel_postnet.transpose(1, 2)


if __name__ == "__main__":
    
    # Test
    checkpoint_in=open(os.path.join(hp.checkpoint_path, 'checkpoint.txt'),'r')
    num=int(checkpoint_in.readline().strip())
    checkpoint_in.close()
    alpha = 1.0
    model = get_FastSpeech(num)
#     condition1=np.load("./data/con1s/000300.npy")
#     condition2=np.load("./data/con2s/000300.npy")
#     D=np.load("./data/alignments/000300.npy")
    
    n=len(os.listdir('./tmp/con1s'))
    
    os.chdir('../FastSinging_') 
    os.system('rm ./tmp/con1s/*')
    os.system('rm ./tmp/con2s/*')
    os.system('rm ./tmp/Ds/*')
    os.system('rm ./tmp/mel/*')
    os.chdir('../FastSinging') 
    
    for index in tqdm(range(n)):
        condition1=np.load("./tmp/con1s/%03d.npy"%index)
        condition2=np.load("./tmp/con2s/%03d.npy"%index)
        D=np.load("./tmp/Ds/%03d.npy"%index)

        mel, mel_postnet, mel_torch, mel_postnet_torch = synthesis(
            model, condition1, condition2, D, alpha=alpha)

        mel=mel_postnet.cpu().numpy()
    #     mel=np.load('./data/mels/000300.npy').astype(np.float32)
#         plt.matshow(mel)
#         plt.savefig('out.png')
    #     plt.cla()

#         print(mel.shape)

        
    #     ap=np.zeros((mel.shape[0],1025)).astype(np.float64)
    #     ap.fill(0.1)

        os.chdir('../FastSinging_') 
        np.save("./tmp/con1s/%03d.npy"%index,condition1)
        np.save("./tmp/con2s/%03d.npy"%index,condition2)
        np.save("./tmp/Ds/%03d.npy"%index,D)
        np.save("./tmp/mels/%03d.npy"%index,mel)
        os.chdir('../FastSinging')

        
    os.chdir('../FastSinging_')        
    os.system('CUDA_VISIBLE_DEVICES=1 python3 synthesis.py')
    os.chdir('../FastSinging') 

    wav=np.zeros(1)

    for index in tqdm(range(n)):
        f0=np.load('../FastSinging_/tmp/f0s/%03d.npy'%index).astype(np.float64)
        f0=f0/5.0+40.0
        f0=440.0*2**((f0-69)/12)

        mel=np.load("../FastSinging_/tmp/mels/%03d.npy"%index)
        arr1=[]
        for i in range(mel.shape[0]):
            arr1.append(np.interp(np.arange(1025),
                         np.linspace(0,1025,128),
                         mel[i][32:])[np.newaxis,:])
        sp=np.concatenate(arr1,axis=0)
    #     plt.matshow(sp)
    #     plt.savefig('out_sp.png')
    #     plt.cla()

        arr2=[]
        for i in range(mel.shape[0]):
            arr2.append(np.interp(np.arange(1025),
                         np.linspace(0,1025,32),
                         mel[i][:32])[np.newaxis,:])
        ap=np.concatenate(arr2,axis=0)
    #     plt.matshow(ap)
    #     plt.savefig('out_ap.png')
        sp=np.exp(sp-1.0)
        ap=(ap+18.0)/20.0
        #     print(ap.max(),ap.min(),ap.mean())

    #     print(f0.shape,sp.shape,ap.shape)
        length=min(f0.shape[0],sp.shape[0],ap.shape[0])
        f0=f0[:length]
        sp=sp[:length]
        ap=ap[:length]
        y = pw.synthesize(f0, sp, ap, 32000, 8.0)
        wav=np.append(wav,y)

    os.chdir('../FastSinging')
    current_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    sf.write('out/out_total_%s.wav'%current_time,wav, 32000)
    # Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
        # "results", str(num) + "_griffin_lim.wav"))

    
    # waveglow.inference.inference(torch.FloatTensor(np.load("/data/mels/001100.npy").T[np.newaxis,:,:]).cuda(), wave_glow, os.path.join(
    #     "results", str(num) + "_waveglow_target.wav"))

    # utils.plot_data([mel.numpy(), mel_postnet.numpy()])

