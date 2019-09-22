# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
语音生成网络模型
参考文献reference/文件夹下
关键点：
梅尔谱
STFT谱
Attention

Acknowledgment：
kyubyong park. kbpark.linguist@gmail.com. 
Arignal Source：
https://www.github.com/kyubyong/tacotron
'''

class Hyperparams:
    '''Hyper parameters'''
    # mode
    sanity_check = True
    
    # data
    text_file = 'data/text.csv'
    sound_fpath = 'data'
    max_len = 100 if not sanity_check else 100 # maximum length of text
    min_len = 10 if not sanity_check else 10 # minimum length of text
    
    # signal processing
    sr = 22050 # Sampling rate. Paper => 24000
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples  This is dependent on the frame_shift.
    win_length = int(sr*frame_length) # samples This is dependent on the frame_length.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 30 # Number of inversion iterations 
    use_log_magnitude = True # if False, use magnitude
    
    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    norm_type = 'ins'  # a normalizer function. value: bn, ln, ins, or None
    
    # training scheme
    lr = 0.0005 # Paper => Exponential decay
    model = "model/TextToSpeech"
    outputdir = 'data/TextToSpeech'
    logdir = "logdir"
    batch_size = 32
    num_epochs = 10000 if not sanity_check else 40 # Paper => 2M global steps!
    loss_type = "l2" # Or you can test "l2"
    num_samples = 32
    
    # etc
    num_gpus = 1 # If you have multiple gpus, adjust this option, and increase the batch size
                 # and run `train_multiple_gpus.py` instead of `train.py`.
    target_zeros_masking = False # If True, we mask zero padding on the target, 
                                 # so exclude them from the loss calculation.     
