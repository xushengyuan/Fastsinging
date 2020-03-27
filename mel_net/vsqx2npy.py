import numpy as np
from pinyin import py2ints
from parsevsqx import vsqx2notes
import sys

def gen_con(begin, end, notes):
    arr = np.zeros([end - begin, 215], dtype=np.float32)

    for i in range(len(notes)):
        nbegin ,nend,_, tone = notes[i]
        if i>0:
            _,_,_,tone_pre=notes[i-1]
        else:
            tone_pre=-1
        if i<len(notes)-1:
            _,_,_,tone_flw=notes[i+1]
        else:
            tone_flw=-1
        # print(note)
        for i in range(nbegin, nend):
            if tone_pre!=-1:
                arr[i - begin][120 + tone_pre] = 1.0
            arr[i - begin][150 + tone] = 1.0   
            if tone_flw!=-1:
                arr[i - begin][180 + tone_flw] = 1.0
            arr[i - begin][210] = (i - nbegin) / (nend - nbegin)
            arr[i - begin][211] = 1 - (i - nbegin) / (nend - nbegin)


    words = []
    i = 0
    while i < len(notes):
        wbegin = notes[i][0]
        word = notes[i][2]
        while i + 1 < len(notes) and notes[i + 1][2] == '-':
            i += 1
            pass
        wend = notes[i][1]
        words.append((wbegin, wend, word))
        i += 1
        pass

    for i in range(len(words)):
        wbegin, wend, word =words[i]
        if i>0:
            _,_,word_pre=words[i-1]
        else:
            word_pre=-1
        if i<len(words)-1:
            _,_,word_flw=words[i+1]
        else:
            word_flw=-1
        for i in range(wbegin, wend):
            if word_pre!=-1:
                ch=0
                c,v=py2ints(word_pre)
                if len(c)!=0:
                    arr[i-begin][c[0]+ch+0]=1.0
                    arr[i-begin][c[1]+ch+9]=1.0
                arr[i-begin][v[0]+ch+20]=1.0
                arr[i-begin][v[1]+ch+36]=1.0
            ch=40
            c,v=py2ints(word)
            if len(c)!=0:
                arr[i-begin][c[0]+ch+0]=1.0
                arr[i-begin][c[1]+ch+9]=1.0
            arr[i-begin][v[0]+ch+20]=1.0
            arr[i-begin][v[1]+ch+36]=1.0
            if word_flw!=-1:
                ch=80
                c,v=py2ints(word_flw)
                if len(c)!=0:
                    arr[i-begin][c[0]+ch+0]=1.0
                    arr[i-begin][c[1]+ch+9]=1.0
                arr[i-begin][v[0]+ch+20]=1.0
                arr[i-begin][v[1]+ch+36]=1.0
            arr[i - begin][212] = (i - wbegin) / (wend - wbegin)
            arr[i - begin][213] = 1 - (i - wbegin) / (wend - wbegin)
            arr[i - begin][214]=1.0

    # print(arr.shape)
    return arr

notes,begin,end = vsqx2notes(sys.argv[1])
con=gen_con(begin,end,notes)
# plt.matshow(con)
# plt.show()
np.save(sys.argv[2],con)
