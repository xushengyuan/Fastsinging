import numpy as np
import os
import pyworld as pw
import soundfile as sf
import Audio
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED, as_completed

dict_size=437
fs = 32000
n=83
# n=10
wav_path="../wav"
tg_path="./data/TextGrid"
dict_path="./pinyin.txt"
mel_ground_truth = "./data/mels"
f0_ground_truth = "./data/f0"
vocoder_con = "./condition"
condition1='./data/con1s'
condition2='./data/con2s'
alignment_path = "./data/alignments"

def time_process(words,phase):
    if phase =='':
        return words
    rate=int(phase[:-1])/100
    _words=[]
    for i in range(len(words)):
        _words.append([int(rate*words[i][0]),int(rate*words[i][1]),words[i][2]])
    return _words

def get_sentences(words):
    sentences=[]
    last=0.0
    for i in range(len(words)-1):
        if words[i][2]=='' and words[i+1][2]!='' :
            last=words[i][1]-int(0.15*fs)
        elif words[i][2]!='' and words[i+1][2]=='' :
            sentences.append((last,words[i][1]+int(0.15*fs),'-'))
    return sentences


def parse_tg(lines):
    lines=lines[7:]
    lines=lines[4:]
    n1=int(lines[0])
    lines=lines[1:]

    for i in range(len(lines)):
        lines[i]=lines[i].strip()
    words=[]
    for i in range(n1):
        words.append([int(fs*float(lines[0])),
                          int(fs*float(lines[1])),
                          lines[2][1:-1]])
        lines=lines[3:]

    sentences=get_sentences(words)

    _words=[]
    for word in words:
        if word[2]!='':
            _words.append(word)

    return _words,sentences    



def collect_words(sentence,words):
    words_in=[]
    # print(words) 
    # print(sentence)
    for word in words:
        if sentence[0]<=word[0] and word[1]<=sentence[1]:
            words_in.append(word)
    # print(words_in)
    return words_in

pinyin={}
def prepare_dict():
    fin=open(dict_path,'r')
    lines=fin.readlines()
    for i in range(len(lines)):
        pinyin[lines[i].strip()]=i+1


def pad_words(words,sentence):
    # print(words)
    # print(sentence)
    if sentence[0]<words[0][0] :
        words=[[sentence[0],words[0][0],'']]+words
    if sentence[1]>words[-1][1]:
        words=words+[[words[-1][1],sentence[1],'']]
    # print(words)
    return words


def word_process(words,sentence):
    for i in range(len(words)):
        words[i][0]=int((words[i][0]-sentence[0])/256+0.5)
        words[i][1]=int((words[i][1]-sentence[0])/256+0.5)
        if words[i][2] in pinyin:
            words[i][2]=pinyin[words[i][2]]
        else:
            words[i][2]=0
    return words


def get_D(words,sentence):
    D=[]
    # print(words)
    for i in range(len(words)):
        length=words[i][1]-words[i][0]
        D.append(int(length))
    return np.array(D)


def get_con1(words):
    # print(words)
    con1=[]
    for i in range(len(words)):
        con1.append(words[i][2])
    return np.array(con1).astype(np.int16)


def herz2note(x):
    return 69+12*np.log(x/440.0)/np.log(2)


def note2herz(n):
    return 440.0*2**((n-69)/12)


def cal_note(_f0):
    f0=[]
    for i in range(len(_f0)):
        if _f0[i]!=0:
            f0.append(_f0[i])
    arr=np.zeros(128,dtype=np.int16)
    for i in range(len(f0)):
        arr[int(f0[i]+0.5)]+=1
    _max=0
    _max_index=0
    for i in range(128):
        if arr[i]>_max:
            _max=arr[i]
            _max_index=i
    if _max_index==127:
        return 0
    return _max_index


def get_con2(x,words):
    _f0, t = pw.dio(x, fs, f0_floor=120.0, f0_ceil=750.0,
                    frame_period=8.0)
    f0_herz = pw.stonemask(x, _f0, t, fs)
    sp = pw.cheaptrick(x, f0_herz, t, fs)
    ap = pw.d4c(x, f0_herz, t, fs)
    # print(sp.shape)

    f0_note=[]
    for i in range(len(f0_herz)):
        if f0_herz[i]==0:
            f0_note.append(0.0)
        else:
            f0_note.append(herz2note(f0_herz[i]))
    con2=[]
    # plt.plot(np.arange(len(x)/256),f0_note)
    for i in range(len(words)):
        note=cal_note(f0_note[words[i][0]:words[i][1]])
        con2.append(note)
        # print(words[i])
        # x=np.arange(words[i][0],words[i][1])
        # y=np.zeros(words[i][1]-words[i][0])
        # y.fill(note)
        # print(x,y)
        # plt.plot(x,y)
    # plt.show()
    f0_note=np.array(f0_note)
    f0_note=np.round((f0_note-40.0)*5)
    # print(f0_note)
    f0_mat=np.zeros([f0_note.shape[0],200])
    f0_mat.fill(0.0)
    for i in range(f0_note.shape[0]):
        if f0_note[i]>0.0 and f0_note[i]<200:
            f0_mat[i][int(f0_note[i])]=1.0
        else:
            f0_note[i]=0

    # plt.matshow(ap)
    # plt.show()
    ap=ap*20-18
    arr=[]
    for i in range(sp.shape[0]):
        arr.append(np.interp(np.linspace(0,1025,32),np.arange(1025),ap[i])[np.newaxis,:])
    _ap=np.concatenate(arr,axis=0)

    sp=np.log(sp)
    # plt.matshow(sp)
    # plt.show()
    arr=[]
    for i in range(sp.shape[0]):
        arr.append(np.interp(np.linspace(0,1025,128),np.arange(1025),sp[i])[np.newaxis,:])
    _sp=np.concatenate(arr,axis=0)

    mel=np.concatenate([_ap,_sp],axis=1)
    
#     mel=mel+20.0
#     mel=np.where(mel>0,mel,0)
#     mel=mel/mel.max()
#     plt.matshow(mel)
#     plt.show()

    return np.array(con2),mel,f0_note.astype(np.int32)


def gen(task):
    sentence,x,words,cnt=task[0],task[1],task[2],task[3]
    # print(words)
    words=pad_words(words,sentence)
    words=word_process(words,sentence)
    D=get_D(words,sentence)
    con1=get_con1(words)    
    con2,mel,f0=get_con2(x,words)

    # print(D)
    # print(mel.shape,D.sum(),D.shape,con1.shape,con2.shape)
    # os.system('pause')

    if D.sum()>mel.shape[0]:
        D[-1]-=D.sum()-mel.shape[0]
    elif D.sum()<mel.shape[0]:
        mel=mel[:D.sum()]

    # print(D)
    # print(mel.shape,D.sum(),D.shape,con1.shape,con2.shape)
    # print(con1)
    # print(con2)
    # os.system('pause')
    if con1.mean()<=0 :
        print(con1,con2,D)
        con1.fill(1)
    assert np.isnan(mel.sum())==False
    assert con1.max()<439 and con1.min()>=0 and con1.mean()>0
    assert con2.max()<128 and con2.min()>=0 and con2.mean()>0
    assert D.shape[0]==con1.shape[0]==con2.shape[0]
    assert mel.shape[0]==D.sum()
    # plt.matshow(mat)
    # plt.show()
    np.save(condition1+"/%06d.npy"%cnt,con1) 
    np.save(condition2+"/%06d.npy"%cnt,con2)
    np.save(mel_ground_truth+"/%06d.npy"%cnt,mel)
    np.save(alignment_path+"/%06d.npy"%cnt,D)
    np.save(f0_ground_truth+"/%06d.npy"%cnt,f0) 
    return 1

def main():
    cnt=0
    executor = ThreadPoolExecutor(max_workers=128)
    all_task=[]    
    tot=0
    cot=0
    for phase1 in ['70_','85_','','125_','150_']:
        for phase2 in ['b2_','b1_','','#1_','#2_']:
            for i in range(1,n+1):
                try:
                    x, _fs = sf.read(wav_path+"/"+phase1+phase2+"%02d.wav"%i)
                    x=x/np.max(np.abs(x))
                    assert _fs==fs
                    tg_in=open(tg_path+"/%02d.TextGrid"%i)
                    tg_lines=tg_in.readlines()
                    words, sentences =parse_tg(tg_lines)
                    words=time_process(words,phase1)
                    sentences=time_process(sentences,phase1)
                    print(phase1,phase2,i)
#                     print(sentences)
                    for sentence in sentences:
                        task=((
                            sentence,
                            x[sentence[0]:sentence[1]],
                            collect_words(sentence,words),
                            cnt
                         ))
                        cnt+=1
                        all_task.append(executor.submit(gen,(task)))
                        tot+=1
                        if tot%2000==0:
                            for future in as_completed(all_task):
                                cot+=future.result()
                                print('\r##',tot,cot,'##',end='')
                            executor = ThreadPoolExecutor(max_workers=128)
                            all_task=[]    
                        # con1=con1[:,np.newaxis]
                        # con2=con2[:,np.newaxis]
                        # D=D[:,np.newaxis]
                        # print(con1.shape)
                        # showmat=np.concatenate([mel/mel.max(),con1/con1.max(),con2/con2.max(),D/D.max()],axis=0)
                        # plt.matshow(showmat)
                        # s=0
                        # for i in range(len(D)):
                            # s+=D[i]
                            # print(s)

                except Exception:
#                     raise
                    print('failed :%d'%i+phase1+phase2)

                # print("finishi %d, total:"%i,cnt)
    
    

prepare_dict()
main()