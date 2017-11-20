# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:40:15 2016

@author: amsturdy
"""

import os
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
import pdb

wav_folder='mivia_db4_original'
wav_class=['background','glass','gunshots','screams']

#pdb.set_trace()

for class_name in wav_class:
    name=[]
    length=[]
    wav_dir=os.path.join(wav_folder,class_name)
    for read_dir,sub_dir,files in os.walk(wav_dir):
        for file in files:
            if(file.endswith('wav')):
                #file_name_no_suffix=os.path.splitext(file)[0]
                full_name=os.path.join(read_dir,file)
                (rate,sig)=wav.read(full_name)
                name.append(full_name)
                length.append(len(sig))
    #pdb.set_trace()
    df1=pd.DataFrame(name,index=np.arange(len(name)),columns=['name'])
    df2=pd.DataFrame(length,index=np.arange(len(length)),columns=['length'])
    df=pd.concat([df1,df2],axis=1)
    sort=df.sort_values(by='length')
    if(os.path.exists(wav_folder+'length_of_'+class_name+'.csv')):
        os.remove(wav_folder+'length_of_'+class_name+'.csv')
    sort.to_csv(wav_folder+'/length_of_'+class_name+'.csv')
