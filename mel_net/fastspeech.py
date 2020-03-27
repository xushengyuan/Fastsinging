import torch
import torch.nn as nn
import os
from Transformer.Models import Encoder, Decoder
from Transformer.Layers import Linear, PostNet
from modules import LengthRegulator
import matplotlib.pyplot as plt
import hparams as hp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder()
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()

        self.mel_linear = Linear(hp.decoder_output_size, hp.num_mels)
        
        self.postnet = PostNet()

    def forward(self, src_seq1, src_seq2, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        
        
        encoder_output, _ = self.encoder(src_seq1, src_seq2 , src_pos)
        
        
        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(encoder_output,target=length_target,alpha=alpha,mel_max_length=mel_max_length)
            decoder_output = self.decoder(length_regulator_output, mel_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output
            
            return mel_output, mel_output_postnet
        else:
#             print(src_seq1,src_seq2,src_pos)
#             print(length_target)
#             plt.matshow(encoder_output[0].cpu().detach().numpy())
#             plt.savefig('out1_encoder_output.png')
            length_regulator_output, decoder_pos = self.length_regulator(encoder_output,target=length_target,alpha=alpha)
#             plt.matshow(length_regulator_output[0].cpu().detach().numpy())
#             plt.savefig('out2_length_regulator_output.png')
            decoder_output = self.decoder(length_regulator_output, decoder_pos)
#             plt.matshow(decoder_output[0].cpu().detach().numpy())
#             plt.savefig('out3_decoder_output.png')
            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output
#             plt.matshow(mel_output[0].cpu().detach().numpy())
#             plt.savefig('out4_mel_output.png')
            # quit()
            return mel_output, mel_output_postnet


if __name__ == "__main__":
    # Test
    model = FastSpeech()
    print(sum(param.numel() for param in model.parameters()))
