# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:40:15 2016

@author: amsturdy
"""

import numpy as np

def dft(x):
    if(len(x.shape)>2):
        print("x's dimension can not exceed 2!")
    if(len(x.shape)==1):
        x=x[np.newaxis,:]	
    n=x.shape[1]
    y=np.zeros((x.shape[0],n,2))
    for row in range(x.shape[0]):
        for i in range(n):
            for j in range(n):
                y[row,i,0]+=x[row,j]*np.cos(-2.0*np.pi*i*j/n)
                y[row,i,1]+=x[row,j]*np.sin(-2.0*np.pi*i*j/n)
    return y

def stfft(x,nfft,noverlap):
    nx=len(x)
    if(nx<nfft):
        apd=np.zeros(nfft-nx,dtype='int16')
        x=np.append(x,apd) #zero-pad x if it has length less than the window length
        nx=nfft
    ncol=int((nx-noverlap)/(nfft-noverlap))
    colindex=np.arange(ncol)*(nfft-noverlap) #the start of every windows
    rowindex=np.arange(nfft)
    if(nx<(nfft+colindex[-1])):
        apd=np.zeros(nfft+colindex[ncol]-nx,dtype='int16') #zero-pad x
        x=np.append(x,apd)
        nx=nfft+colindex[ncol]
    loc=np.tile(colindex,(nfft,1)).T+np.tile(rowindex,(ncol,1))
    hammings=np.tile(np.hamming(nfft),(ncol,1))
    x=x[loc]*hammings
    
    xf=np.fft.fft(x)
    if(nfft/2==0):
        select=xf[:,:nfft/2]
    else:
        select=xf[:,:(nfft+1)/2]
    im=20*np.log10(np.abs(select))
    '''
    xf=dft(x)
    if(nfft/2==0):
        select=xf[:,0:(nfft/2),:]
    else:
        select=xf[:,0:(nfft+1)/2,:]
    im=np.zeros((ncol,select.shape[1]))
    for i in range(select.shape[1]):
        im[:,i]=20*np.log10(pow(pow(select[:,i,0],2)+pow(select[:,i,1],2),0.5))
    '''
    return im

if __name__ == '__main__':
    import scipy.io.wavfile as wav
    from xml.etree import ElementTree as ET
    import matplotlib.pyplot as plt
    from PIL import Image

    nfft=511
    noverlap=256

    (rate,sig)=wav.read('mivia_db4/training/sounds/00001_1.wav')
    per = ET.parse('mivia_db4/training/00001.xml')
    events = per.findall('./events/item')

    interest_start=int(rate*float(events[0].findall('STARTSECOND')[0].text))
    interest_end=int(rate*float(events[0].findall('ENDSECOND')[0].text))

    START=0
    END=int(rate*float(events[1].findall('STARTSECOND')[0].text))
	    
    picture_interest_start=int((interest_start-START)/noverlap)
    picture_interest_end=int((interest_end-START)/noverlap+1)

    x=np.array(sig[START:END])
    im=stfft(x,nfft,noverlap)
    image=Image.fromarray(im.astype(np.uint8))
    image.save('test.jpg')

    image=np.array(image)
    im_size=im.shape
    fig, ax = plt.subplots(figsize=(im_size[1]/100.0, im_size[0]/100.0))
    ax.imshow(im, aspect='equal')
    ax.add_patch(
	        plt.Rectangle((0, picture_interest_start),
	              256,picture_interest_end-picture_interest_start,
		      fill=False,edgecolor='red', linewidth=3.5)
	    )
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig('test1.jpg')
    plt.close()
