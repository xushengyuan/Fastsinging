import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams
import Audio
from text import text_to_sequence
from utils import pad_1D, pad_2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeechDataset(Dataset):
    """ LJSpeech """

    def __init__(self):
        mel_n=len(os.listdir(hparams.mel_ground_truth))
        mel_in_n=len(os.listdir(hparams.mel_in))
        c1_n=len(os.listdir(hparams.condition1))
        c2_n=len(os.listdir(hparams.condition2))
        assert mel_n==c1_n,c1_n==c2_n==mel_in_n
        self.n=c1_n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        assert idx<self.n
        mel_in_name = os.path.join(hparams.mel_in,"%06d.npy"%(idx))
        mel_in=np.load(mel_in_name)
        mel_gt_name = os.path.join(
            hparams.mel_ground_truth, "%06d.npy" % (idx))
        mel_gt_target = np.load(mel_gt_name).astype(np.int32)

        
           
        D = np.load(os.path.join(hparams.alignment_path, "%06d.npy"%idx))
        
        C1_name = os.path.join(hparams.condition1,"%06d.npy"%(idx))
        C1=np.load(C1_name)
        C1=np.where(C1<0,0,C1)
        C1=np.where(C1>=439,0,C1)
        C2_name = os.path.join(hparams.condition2,"%06d.npy"%(idx))
        C2=np.load(C2_name)
        C2=np.where(C2<0,0,C2)
        C2=np.where(C2>=128,0,C2)
        
        norm_f0=np.zeros(mel_gt_target.shape[0]).astype(np.int32)
#         print(mel_gt_target.shape,D.shape,C2.shape,norm_f0.shape)
        for i in range(C2.shape[0]):
            for j in range(D[i]):
                if mel_gt_target[D[:i].sum()+j]!=0:
                    norm_f0[D[:i].sum()+j]=(C2[i]-40)*5
        norm_f0=np.where(norm_f0<0,0,norm_f0)
        norm_f0=np.where(norm_f0<0,0,norm_f0)
        norm_f0=np.where(norm_f0>=200,0,norm_f0)
#         print(norm_f0)
        
        
        norm_f0=norm_f0[:,np.newaxis]
        mel_gt_target=mel_gt_target[:,np.newaxis]
        
        assert norm_f0.shape==mel_gt_target.shape
        sample = {"condition1": C1,
                  "condition2":C2,
                  "mel_target": mel_gt_target,
                  "norm_f0":norm_f0,
                  "mel_in":mel_in,
                  "D": D}
        return sample


def reprocess(batch, cut_list):
    C1s = [batch[ind]["condition1"] for ind in cut_list]
    C2s = [batch[ind]["condition2"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]    
    norm_f0s = [batch[ind]["norm_f0"] for ind in cut_list]
    Ds = [batch[ind]["D"] for ind in cut_list]
    mel_ins = [batch[ind]["mel_in"] for ind in cut_list]

    length_C = np.array([])
    for C in C1s:
        length_C = np.append(length_C, C.shape[0])

    src_pos = list()
    max_len = int(max(length_C))
    for length_src_row in length_C:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = np.array(src_pos)

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    mel_pos = list()
    lens = torch.LongTensor(length_mel)
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = np.array(mel_pos)

    C1s = pad_1D(C1s)
    C2s = pad_1D(C2s)
    Ds = pad_1D(Ds)
    norm_f0s = pad_2D(norm_f0s,maxlen=max_mel_len)
    mel_targets = pad_2D(mel_targets,maxlen=max_mel_len)
    mel_ins = pad_2D(mel_ins,maxlen=max_mel_len)

    out = {"condition1": C1s,
           "condition2":C2s,
           "mel_target": mel_targets,
           "norm_f0":norm_f0s,
           "mel_in": mel_ins,
           "D": Ds,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "lens": lens,
           "mel_max_len": max_mel_len}

    return out


def collate_fn(batch):
    len_arr = np.array([d["condition1"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(reprocess(batch, cut_list[i]))

    return output


if __name__ == "__main__":
    # Test
    dataset = FastSpeechDataset()
    training_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 drop_last=True,
                                 num_workers=0)
    total_step = hparams.epochs * len(training_loader) * hparams.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            # print(mel_target.size())
            # print(D.sum())
            print(cnt)
            if mel_target.size(1) == D.sum().item():
                cnt += 1

    print(cnt)
