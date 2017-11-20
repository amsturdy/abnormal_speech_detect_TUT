# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:40:15 2016

@author: amsturdy
"""
from mystfft import stfft
import yaml
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.io.wavfile as wav
import numpy as np
import shutil

CLASS_NAMES=['babycry','glassbreak','gunshot']
num_of_every_class=5000
EBR={-6.0:0,0.0:1,6.0:2}
common_fs=44100
nfft=1024
noverlap=nfft-441

cache_file='py-R-FCN/data/cache/voc_0712_trainval_gt_roidb.pkl'
if(os.path.exists(cache_file)):
    os.remove(cache_file)

audio_folder='TUT-rare-sound-events-2017-development/data/mixture_data/devtrain/625ce518d46178f3134a6f8b3716da03/'
data_folder='py-R-FCN/data/VOCdevkit0712/VOC0712/'
if(os.path.exists(data_folder)):
    shutil.rmtree(data_folder)
os.makedirs(data_folder+'JPEGImages')
os.makedirs(data_folder+'ImageSets/Main')
os.makedirs(data_folder+'Labels')

def read_meta_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.load(infile)
    return data

def load_audio(path, target_fs=None):
    """
    Reads audio with (currently only one supported) backend (as opposed to librosa, supports more formats and 32 bit
    wavs) and resamples it if needed
    :param path: path to wav/flac/ogg etc (http://www.mega-nerd.com/libsndfile/#Features)
    :param target_fs: if None, original fs is kept, otherwise resampled
    :return:
    """
    y, fs = sf.read(path)
    if y.ndim>1:
        y = np.mean(y, axis=1)
    if target_fs is not None and fs!=target_fs:
        #print('Resampling %d->%d...' %(fs, target_fs))
        y = librosa.resample(y, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return y, fs

if __name__ == '__main__':
    audio=audio_folder+'audio/'

    babycry_yaml=audio_folder+'meta/mixture_recipes_devtrain_babycry.yaml'
    glassbreak_yaml=audio_folder+'meta/mixture_recipes_devtrain_glassbreak.yaml'
    gunshot_yaml=audio_folder+'meta/mixture_recipes_devtrain_gunshot.yaml'
    
    picture_txt=data_folder+'picture.txt'
    fpicture=open(picture_txt,'wt')

    trainval_txt=data_folder+'ImageSets/Main/trainval.txt'
    ftrainval=open(trainval_txt,'wt')

    train_txt=data_folder+'ImageSets/Main/train.txt'
    ftrain=open(train_txt,'wt')

    val_txt=data_folder+'ImageSets/Main/val.txt'
    fval=open(val_txt,'wt')

    for i,class_yaml in enumerate([babycry_yaml,glassbreak_yaml,gunshot_yaml]):
        data=read_meta_yaml(class_yaml)
	image_num=0
	for item in data:
            image_num+=1
	    full_audio_name=audio+item['mixture_audio_filename']
	    if os.path.exists(full_audio_name):
	        x,rate=load_audio(full_audio_name,common_fs)
	    else:
		continue
	    picture_interest_start=int(rate*item['event_start_in_mixture_seconds']/(nfft-noverlap))
            picture_interest_end=int(rate*(item['event_start_in_mixture_seconds']+item['event_length_seconds'])/(nfft-noverlap))

            im=stfft(x,nfft,noverlap)
            image=Image.fromarray(im.astype(np.uint8))
	    image_name='IMG_'+str(i)+str(EBR[item['ebr']])+"%04d"%(image_num)
	    image.save(data_folder+'JPEGImages/'+image_name+'.jpg')
            fpicture.write(image_name+'.jpg '+full_audio_name+'\n')
            ftrainval.write(image_name+'\n')
            if(image_num<int(num_of_every_class*0.8)):
   	        ftrain.write(image_name+'\n')
	    else:
		fval.write(image_name+'\n')
   	    label_txt=data_folder+'Labels/'+image_name+'.txt'
	    flabel=open(label_txt,'wt')
	    flabel.write(image_name+'\n' \
			+str(image.size[0])+' '+str(image.size[1])+' 1\n')
	    flabel.write(CLASS_NAMES[i]+'\n' \
			+str(1)+' '+str(picture_interest_start)+' '+str(image.size[0])+' '+str(picture_interest_end)+'\n')
	    flabel.close
    ftrainval.close
    ftrain.close
    fval.close
    fpicture.close
