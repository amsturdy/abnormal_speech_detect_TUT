# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:40:15 2016

@author: amsturdy
"""
import h5py
import stfft
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.io.wavfile as wav
import numpy as np
import shutil

singnal_to_noise_ratio=['_01.wav','_00.wav','_1.wav','_2.wav','_3.wav','_4.wav','_5.wav','_6.wav']
nfft=511
noverlap=256
name_list=['training','testing']

data_folder='net/data'
if(os.path.exists(data_folder)):
    shutil.rmtree(data_folder)
os.makedirs(data_folder)

for name in name_list:
    if name=='testing':
	break

    xml_folder='mivia_db4/'+name
    sounds_folder=xml_folder+'/sounds'
    data=[]
    label=[]
    for i,file in enumerate(sorted(os.listdir(xml_folder))):
        if(file.endswith('xml')):
            file_name_no_suffix=os.path.splitext(file)[0]
            singnal=[]
            for suffix in singnal_to_noise_ratio:
                wav_file_name=file_name_no_suffix+suffix
                wav_full_name=os.path.join(sounds_folder,file_name_no_suffix+suffix)
                (rate,sig)=wav.read(wav_full_name)
                singnal.append(sig)
            xml_full_name=os.path.join(xml_folder,file)
            per = ET.parse(xml_full_name)
            events = per.findall('./events/item')
            for j in np.arange(len(events)):
	        if(j==0):
	   	    START=0
	        else:
        	    START=int(rate*float(events[j-1].findall('ENDSECOND')[0].text))
                if(j==(len(events)-1)):
		    END=-1
	        else:
		    END=int(rate*float(events[j+1].findall('STARTSECOND')[0].text))
	    
	        for k,suffix in enumerate(singnal_to_noise_ratio):
                    x=np.array(singnal[-1][START:END])
		    if(name=='training'):
		        data.append(x[np.newaxis,np.newaxis,:])
		        im=stfft.stfft(x,nfft,noverlap)
                        label.append(im.T[np.newaxis,:])
		    else:
		        data.append(x[np.newaxis,np.newaxis,:])		    
    
    h5_filename = data_folder+'/{}.h5'.format(name)
    with h5py.File(h5_filename, 'w') as h:        
        h.create_dataset('data', data=np.array(data))
	if(name=='training'):
            h.create_dataset('label', data=np.array(label))

    with open(data_folder+'/{}_h5.txt'.format(name), 'w') as f:
        f.write(h5_filename)

    with h5py.File(h5_filename, 'r') as h:        
        data=h['data']
	label=h['label']
