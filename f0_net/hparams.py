from text import symbols

# Text
text_cleaners = ['english_cleaners']

# Mel
n_mel_channels = 200
num_mels = 200
mel_in_size=160

# FastSpeech
vocab_size = 439 
note_size = 128
N = 4
Head = 2
d_model = 128
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

word_vec_dim = 128
encoder_n_layer = 6
encoder_head = 2
encoder_conv1d_filter_size = 1536
max_sep_len = 4096
encoder_output_size = 128
decoder_n_layer = 6
decoder_head = 2
decoder_conv1d_filter_size = 1536
decoder_output_size = 128
fft_conv1d_kernel = 3
fft_conv1d_padding = 1
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

# Train
alignment_path = "./data/alignments"
checkpoint_path = "./model_new"
logger_path = "./logger"
mel_ground_truth = "./data/f0"
mel_in = "./data/mels"
condition1='./data/con1s'
condition2='./data/con2s'

batch_size = 4
epochs = 1000
n_warm_up_step = 2000

learning_rate = 1e-4
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 5000
log_step = 10
gpu_log_step = -1
clear_Time = 200
fig_step = 1000