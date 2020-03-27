import xml.dom.minidom
import random

def vsqx2notes(path):
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    trackElements=root.getElementsByTagName('vsTrack')
    masterElement=root.getElementsByTagName('masterTrack')[0]
    resolutionElement=masterElement.getElementsByTagName('resolution')[0]
    resolution=int(resolutionElement.childNodes[0].data)
    preElement=masterElement.getElementsByTagName('preMeasure')[0]
    pre=int(preElement.childNodes[0].data)
    timeSigElement=masterElement.getElementsByTagName('timeSig')[0]
    timeSig=(int(timeSigElement.getElementsByTagName('nu')[0].childNodes[0].data),
                int(timeSigElement.getElementsByTagName('de')[0].childNodes[0].data))
    tempoElement=masterElement.getElementsByTagName('tempo')[0]
    tempo=int(tempoElement.getElementsByTagName('v')[0].childNodes[0].data)
    mspt=(60000.0/tempo*100)/resolution
    jj=0
    for trackElement in trackElements:
        track=[]
        partElements=trackElement.getElementsByTagName('vsPart')
        for partElement in partElements:
            pt=int(partElement.getElementsByTagName('t')[0].childNodes[0].data)
            noteElements=partElement.getElementsByTagName('note')
            for noteElement in noteElements:
                tElement=noteElement.getElementsByTagName('t')[0]
                t=int(tElement.childNodes[0].data)
                durElement=noteElement.getElementsByTagName('dur')[0]
                dur=int(durElement.childNodes[0].data)
                nElement=noteElement.getElementsByTagName('n')[0]
                n=int(nElement.childNodes[0].data)
                lrcElement=noteElement.getElementsByTagName('y')[0]
                lrc=lrcElement.childNodes[0].data.lower()
                ipaElement=noteElement.getElementsByTagName('p')[0]
                ipa=ipaElement.childNodes[0].data
                track.append((pt+t,dur,n,lrc,ipa))
        # last=0
        # lastn=0
        # track.append((0,0,0,'',''))
        # for i in range(len(track)-1):
        #     #print(note)
        #     note=track[i]
        #     for j in range(int((note[0]-last)*mspt/10)):
        #         fout.write('0\n')
        #     rest=int((note[0]-last)*mspt)-int((note[0]-last)*mspt/10)*10
        #     for j in range(int((note[1]) * mspt / 10 / 4 * 5)):
        #         fout.write('%d %d %d %f '%(track[i+1][2],note[2],lastn,j/((note[1])*mspt/10)))
        #         try:
        #             fout.write('%d '%dicty[note[3].strip()])
        #         except KeyError:
        #             print('KeyError %s'%note[3].strip())
        #         fout.write('\n')
        #     rest=int((note[1])*mspt)-int((note[1])*mspt/10)*10
        #     last=note[0]+note[1]
        #     lastn=note[2]
        # jj+=1
        # fout.close()
        notes=[]
        end=0
        begin=23333333333333333333333
        for note in track:
            notes.append((int(note[0]*mspt/8),#begin
            int((note[0]+note[1])*mspt/8),#end
            note[3],
            note[2]))
            begin=min(begin,int(note[0]*mspt/8))
            end = max(end, int((note[0] + note[1]) * mspt / 8))
        return notes,begin,end