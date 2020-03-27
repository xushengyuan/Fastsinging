import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
from parsevsqx import vsqx2notes
import random
import shutil
from subprocess import Popen

dict_size=437
fs = 32000
dict_path="./pinyin.txt"

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
        words=[[sentence[0],words[0][0],'',64]]+words
    if sentence[1]>words[-1][1]:
        words=words+[[words[-1][1],sentence[1],'',64]]
    # print(words)
    return words
        
def get_D(words):
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
        if words[i][2] in pinyin:
            con1.append(pinyin[words[i][2]])
        else:
            con1.append(0)
    return np.array(con1)

def get_con2(words):
    con2=[]
    for i in range(len(words)):
        con2.append(words[i][3])
    return np.array(con2)

def gen(notes,sentence):
    notes=pad_words(notes,sentence)
    D=get_D(notes)
    con1=get_con1(notes)    
    con2=get_con2(notes)

    # print(D)
    # print(mel.shape,D.sum(),D.shape,con1.shape,con2.shape)
    # os.system('pause')


    # print(D)
    # print(mel.shape,D.sum(),D.shape,con1.shape,con2.shape)
    # print(con1)
    # print(con2)
    # os.system('pause')
    assert D.shape[0]==con1.shape[0]==con2.shape[0]
    # assert mel.shape[0]==D.sum()
    return [con1,con2,D]

def process_D(D):
    for i in range(len(D)):
        D[i]=int(D[i]+6*random.random()-3)
    return D

prepare_dict()
# main()

words,begin,end = vsqx2notes(sys.argv[1])

wav=np.zeros(1)

length1=20
last=begin
last_n=0
cot=1
i=0
con1s=[]
con2s=[]
Ds=[]
while i <len(words)-1:
    if words[i][1]!=words[i+1][0]:
        
        length2=words[i+1][0]-words[i][1]
        
        begin=last-10
        end=words[i][1]+length2
        
        length1=length2
        # print(begin,end)
        print('Part %d: from %d to %d len:%d n_note: %d'%(cot,begin,end,end-begin,i+1-last_n))
        con1,con2,D=gen(words[last_n:i+1],(begin,end,'-'))

        # print(con1)
        # print(con2)
        # print(D)

        D=process_D(D)
        # print(D)
        con1s.append(con1)
        con2s.append(con2)
        Ds.append(D)
        
        last=words[i+1][0]
        last_n=i+1
        cot+=1
    i+=1
    

length2=40
        
begin=last-10
end=words[i][1]+length2
        
length1=length2
print('Part %d: from %d to %d len:%d n_note: %d'%(cot,begin,end,end-begin,i+1-last_n))
con1,con2,D=gen(words[last_n:i+1],(begin,end,'-'))

D=process_D(D)
# print(D)
con1s.append(con1)
con2s.append(con2)
Ds.append(D)

os.chdir('./f0_net') 
shutil.rmtree('tmp')
os.mkdir('tmp')
os.mkdir('tmp/con1s')
os.mkdir('tmp/con2s')
os.mkdir('tmp/Ds')
os.mkdir('tmp/mels')
os.mkdir('tmp/f0s')

os.chdir('../mel_net')
shutil.rmtree('tmp')
os.mkdir('tmp')
os.mkdir('tmp/con1s')
os.mkdir('tmp/con2s')
os.mkdir('tmp/Ds')


for i in range(len(con1s)):
    np.save('./tmp/con1s/%03d.npy'%i,con1s[i])
    np.save('./tmp/con2s/%03d.npy'%i,con2s[i])
    np.save('./tmp/Ds/%03d.npy'%i,Ds[i])
   

p = Popen('python synthesis.py')
p.wait()