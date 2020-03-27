import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams
from text import text_to_sequence
from utils import pad_1D, pad_2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeechDataset(Dataset):
    """ LJSpeech """

    def __init__(self):
        mel_n=len(os.listdir(hparams.mel_ground_truth))
        c1_n=len(os.listdir(hparams.condition1))
        c2_n=len(os.listdir(hparams.condition2))
        assert mel_n==c1_n,c1_n==c2_n
        self.n=c1_n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        assert idx<self.n
        mel_gt_name = os.path.join(
            hparams.mel_ground_truth, "%06d.npy" % (idx))
        mel_gt_target = np.load(mel_gt_name)

        D = np.load(os.path.join(hparams.alignment_path, "%06d.npy"%idx))

        C1_name = os.path.join(hparams.condition1,"%06d.npy"%(idx))
        C1=np.load(C1_name)
        C2_name = os.path.join(hparams.condition2,"%06d.npy"%(idx))
        C2=np.load(C2_name)
        
        sample = {"condition1": C1,
                  "condition2":C2,
                  "mel_target": mel_gt_target,
                  "D": D}
        return sample


def reprocess(batch, cut_list):
    C1s = [batch[ind]["condition1"] for ind in cut_list]
    C2s = [batch[ind]["condition2"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    Ds = [batch[ind]["D"] for ind in cut_list]

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
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = np.array(mel_pos)

    C1s = pad_1D(C1s)
    C2s = pad_1D(C2s)
    Ds = pad_1D(Ds)
    mel_targets = pad_2D(mel_targets)

    out = {"condition1": C1s,
           "condition2":C2s,
           "mel_target": mel_targets,
           "D": Ds,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
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
