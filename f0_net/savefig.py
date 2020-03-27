import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
mel_in=open('./logger/mel_loss.txt')
mel_pn_in=open('./logger/mel_postnet_loss.txt')
total_in=open('./logger/total_loss.txt')

mel_loss=np.array([float(line.strip()) for line in mel_in])
mel_pn_loss=np.array([float(line.strip()) for line in mel_pn_in])
total_loss=np.array([float(line.strip()) for line in total_in])

l=min(len(mel_loss),len(mel_pn_loss),len(total_loss))
mel_loss=mel_loss[:l-1]
mel_pn_loss=mel_pn_loss[:l-1]
total_loss=total_loss[:l-1]

plt.plot(np.arange(mel_loss.shape[0]),mel_loss,linewidth = '1')
plt.plot(np.arange(mel_pn_loss.shape[0]),mel_pn_loss,linewidth = '1')
plt.plot(np.arange(total_loss.shape[0]),total_loss,linewidth = '1')

length=200

def plot_smooth(total_loss):
  total_loss=total_loss[:length*int(total_loss.shape[0]/length)]
  mean_loss=[]
  for i in range(int(total_loss.shape[0]/length)):
    mean_loss.append(total_loss[i*length:(i+1)*length].mean())
  mean_loss=np.array(mean_loss)
  x=np.arange(length/2,total_loss.shape[0]+length/2,length)
  # print(x.shape,mean_loss.shape)
  loss_smooth = interp1d(x,mean_loss,kind='cubic')(np.arange(length,(mean_loss.shape[0]-1)*length))
  plt.plot(np.arange(loss_smooth.shape[0]),loss_smooth)

plot_smooth(mel_loss)
plot_smooth(mel_pn_loss)
plot_smooth(total_loss)
plt.savefig('loss.png')