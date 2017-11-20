import _init_paths
import caffe
import numpy as np
import scipy.io.wavfile as wav
from mystfft import stfft
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc

nfft=511
noverlap=256

(rate,sig)=wav.read('mivia_db4/training/sounds/00001_1.wav')
per = ET.parse('mivia_db4/training/00001.xml')
events = per.findall('./events/item')

interest_start=int(rate*float(events[0].findall('STARTSECOND')[0].text))
interest_end=int(rate*float(events[0].findall('ENDSECOND')[0].text))

START=0
END=int(rate*float(events[1].findall('STARTSECOND')[0].text))
    
picture_interest_start=int((interest_start-START)/noverlap-1)
picture_interest_end=int((interest_end-START)/noverlap+1)

x=np.array(sig[START:END])
x=x[np.newaxis,np.newaxis,np.newaxis,:]

caffe.set_mode_cpu()

#caffe.set_device(0)  # if we have multiple GPUs, pick the first one
#caffe.set_mode_gpu()

model_def = 'net/deploy.prototxt'
model_weights = 'net/pretraining.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)

#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
net.blobs['data'].reshape(*(x.shape))
net.blobs['data'].data[...] = x

output = net.forward()['output']
im=np.array(output[0,0,:,:]).T
misc.imsave('test.jpg', im)
#image=Image.fromarray(im.astype(np.uint8))
#image.save('test.jpg')

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
