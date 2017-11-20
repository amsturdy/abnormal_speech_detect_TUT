# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:40:15 2016

@author: amsturdy
"""

from xml.etree import ElementTree as ET
import os
import numpy as np
import pandas as pd
import pdb

name='training'
xml_folder='mivia_db4/'+name

#pdb.set_trace()

ID=[]
NAME=[]
LENGTH=[]
for file in sorted(os.listdir(xml_folder)):
    if(file.endswith('xml')):
        xml_full_name=os.path.join(xml_folder,file)
        per = ET.parse(xml_full_name)
        events = per.findall('./events/item')
        backgrounds = per.findall('./background/item')
        path_name='_'
        for background in backgrounds:
            path_name+=('_'+background.findall('PATHNAME')[0].text)
        background_start=0
        for i,event in enumerate(events):
            class_name=event.findall('CLASS_NAME')[0].text
            class_id=int(event.findall('CLASS_ID')[0].text)
            start_second=int(32000*float(event.findall('STARTSECOND')[0].text))
            end_second=int(32000*float(event.findall('ENDSECOND')[0].text))
            ID.append(0)
            NAME.append(path_name)
            LENGTH.append(start_second-background_start)
            ID.append(class_id-1)
            NAME.append(class_name)
            LENGTH.append(end_second-start_second)
            background_start=end_second
        #pdb.set_trace()
#pdb.set_trace()
df1=pd.DataFrame(ID,index=np.arange(len(ID)),columns=['id'])
df2=pd.DataFrame(NAME,index=np.arange(len(NAME)),columns=['name'])
df3=pd.DataFrame(LENGTH,index=np.arange(len(LENGTH)),columns=['length'])
df=pd.concat([df1,df2,df3],axis=1)
sort=df.sort_values(by='id')

txt='mivia_db4/'+name+'.csv'
if(os.path.exists(txt)):
    os.remove(txt)

sort.to_csv(txt)
