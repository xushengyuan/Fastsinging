import torch
import torch.nn as nn

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time
import math
import matplotlib.pyplot as plt

from fastspeech import FastSpeech
from loss import FastSpeechLoss
from dataset import FastSpeechDataset, collate_fn, DataLoader
import hparams as hp
import utils

from tensorboardX import SummaryWriter

def main(args):
    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
#     torch.distributed.init_process_group(backend='nccl')
    
    # Define model
    model = nn.DataParallel(FastSpeech())
    model=model.cuda()
    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of FastSpeech Parameters:', num_param)


    
    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    writer = SummaryWriter(log_dir='log/'+current_time)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

    # Load checkpoint if exists
    try:
        checkpoint_in=open(os.path.join(hp.checkpoint_path, 'checkpoint.txt'),'r')
        args.restore_step=int(checkpoint_in.readline().strip())
        checkpoint_in.close()
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path,  'checkpoint_%08d.pth'%args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)
    # Get dataset
    dataset = FastSpeechDataset()

    # Optimizer and loss
    
    
    fastspeech_loss = FastSpeechLoss().to(device)
    print("Defined Optimizer and Loss Function.")

    

    # Init logger
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Define Some Information
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    
#     model = torch.nn.parallel.DistributedDataParallel(model) # device_ids will include all GPU devices by default
    print('Start')
#     model = model.train()
    
    for epoch in range(hp.epochs):
        # Get Training Loader
        print('Start Epoch %d'%epoch)
        training_loader = DataLoader(dataset,
                                     batch_size=hp.batch_size**2,
                                     shuffle=True,
                                     collate_fn=collate_fn,
                                     drop_last=True,
                                     num_workers=0)
        total_step = hp.epochs * len(training_loader) * hp.batch_size
        
        m_l=0.0
        m_p_l=0.0
        t_l=0.0
        for i, batchs in enumerate(training_loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.perf_counter()

                current_step = i * hp.batch_size + j + args.restore_step + \
                    epoch * len(training_loader)*hp.batch_size + 1

                model.zero_grad()

                # Get Data
                condition1 = torch.from_numpy(
                    data_of_batch["condition1"]).long().to(device)#.fill_(1)
                condition2 = torch.from_numpy(
                    data_of_batch["condition2"]).long().to(device)#.fill_(1)
                mel_target = torch.from_numpy(
                    data_of_batch["mel_target"]).float().to(device)
                D = torch.from_numpy(data_of_batch["D"]).int().to(device)
                mel_pos = torch.from_numpy(
                    data_of_batch["mel_pos"]).long().to(device)
                src_pos = torch.from_numpy(
                    data_of_batch["src_pos"]).long().to(device)
                max_mel_len = data_of_batch["mel_max_len"]

                # Forward
                mel_output, mel_postnet_output = model(src_seq1=condition1,src_seq2=condition2,
                                                                                  src_pos=src_pos,
                                                                                  mel_pos=mel_pos,
                                                                                  mel_max_length=max_mel_len,
                                                                                  length_target=D)

#                 print(mel_target.size())
#                 print(mel_output)
#                 print(mel_postnet_output)

                # Cal Loss
                mel_loss, mel_postnet_loss= fastspeech_loss(mel_output,                                                                            mel_postnet_output,mel_target,)
                total_loss =mel_loss+mel_postnet_loss 

                # Logger
                t_l += np.log(total_loss.item())
                m_l += np.log(mel_loss.item())
                m_p_l += np.log(mel_postnet_loss.item())
                
#                 assert np.isnan(t_l)==False

                with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")

                with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")

                with open(os.path.join("logger", "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l)+"\n")

                # Backward
                if not np.isnan(t_l):
                    total_loss.backward()
                    optimizer.step()
                else:
                    print(condition1,condition2,D)

                # Clipping gradients to avoid gradient explosion
#                 nn.utils.clip_grad_norm_(
#                     model.parameters(), hp.grad_clip_thresh)

                # Update weights
#                 if args.frozen_learning_rate:
#                     scheduled_optim.step_and_update_lr_frozen(
#                         args.learning_rate_frozen)
#                 else:
#                     scheduled_optim.step_and_update_lr()


                # Print
                if current_step % hp.log_step == 0:
                    Now = time.perf_counter()
                    
                    str1 = "Epoch[{}/{}] Step[{}/{}]:".format(
                        epoch+1, hp.epochs, current_step, total_step)
                    str2 = "Mel Loss:{:.4f} MelPostNet Loss:{:.4f}".format(
                        m_l/hp.log_step, m_p_l/hp.log_step)
                    str3 = "LR:{:.6f}".format(
                        hp.learning_rate)
                    str4 = "T: {:.1f}s ETR:{:.1f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))
                    
                    writer.add_scalar('Mel Loss', m_l/hp.log_step, current_step)
                    writer.add_scalar('MelPostNet Loss', m_p_l/hp.log_step, current_step)
                    writer.add_scalar('Loss', t_l/hp.log_step, current_step)
                    writer.add_scalar('lreaning rate', hp.learning_rate, current_step)
                    
                    
                    print('\r' + str1+' '+str2+' '+str3+' '+str4,end='')

                    if hp.gpu_log_step!=-1 and current_step%hp.gpu_log_step==0:
                        os.system('nvidia-smi')

                    with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write(str4 + "\n")
                        f_logger.write("\n")
                    m_l=0.0
                    m_p_l=0.0
                    t_l=0.0
                if current_step % hp.fig_step==0:
                  f=plt.figure()
                  plt.matshow(mel_postnet_output[0].cpu().detach().numpy())
                  plt.savefig('out_predicted_postnet.png')
                  writer.add_figure('predict',f,current_step)
                  plt.cla() 
                  f=plt.figure()
                  plt.matshow(mel_target[0].cpu().detach().numpy())
                  plt.savefig('out_target.png')
                  writer.add_figure('target',f,current_step)
                  plt.cla() 
                  plt.close("all")
                  

                if current_step % (hp.save_step) == 0:
                    print("save model at step %d ..." % current_step,end='')
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(hp.checkpoint_path, 'checkpoint_%08d.pth'%current_step))
                    checkpoint_out=open(os.path.join(hp.checkpoint_path, 'checkpoint.txt'),'w')
                    checkpoint_out.write(str(current_step))
                    checkpoint_out.close()

                    

#                     os.system('python savefig.py')

                end_time = time.perf_counter()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=False)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-3)
    args = parser.parse_args()

    main(args)
