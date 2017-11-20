# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:40:15 2016

@author: amsturdy
"""
import stfft
import yaml
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.io.wavfile as wav
import numpy as np
import shutil

def read_meta_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.load(infile)
    return data

if __name__ == '__main__':
    for phase in ['devtrain','devtest']:
	if phase=='devtrain':
	    audio_folder='TUT-rare-sound-events-2017-development/data/mixture_data/devtrain/625ce518d46178f3134a6f8b3716da03/'
	else:	
	    audio_folder='TUT-rare-sound-events-2017-development/data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/'

        audio=audio_folder+'audio/'

        babycry_yaml=audio_folder+'meta/mixture_recipes_'+phase+'_babycry.yaml'
        glassbreak_yaml=audio_folder+'meta/mixture_recipes_'+phase+'_glassbreak.yaml'
        gunshot_yaml=audio_folder+'meta/mixture_recipes_'+phase+'_gunshot.yaml'
    
        lengths=[]
        for i,class_yaml in enumerate([babycry_yaml,glassbreak_yaml,gunshot_yaml]):
            data=read_meta_yaml(class_yaml)
	    for item in data:
		if item.has_key('event_length_seconds'):
                    lengths.append(item['event_length_seconds'])

	print max(lengths),min(lengths)    
        #np.save(audio_folder+pahse+'_interest_lengths.npy',np.array(lengths))
