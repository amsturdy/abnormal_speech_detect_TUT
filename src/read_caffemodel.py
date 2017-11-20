import _init_paths
import caffe
import numpy as np

caffe.set_mode_cpu()

#caffe.set_device(0)  # if we have multiple GPUs, pick the first one
#caffe.set_mode_gpu()

model_def = 'net/net.prototxt'
model_weights = 'net/pretraining.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST) 

#for layer_name, blob in net.blobs.iteritems():
#    print layer_name + '\t' + str(blob.data.shape)

#for layer_name, param in net.params.iteritems():
#    for i in range(len(param)):
#    	print layer_name + '\t' + str(param[i].data.shape)

hamming=np.hamming(511)

for i in range(256):
    for j in range(511):
	net.params['conv1'][0].data[i,0,0,j]=np.cos(-2.0*np.pi*i*j/511)*hamming[j]

for i in range(256):
    for j in range(511):
	net.params['conv2'][0].data[i,0,0,j]=np.sin(-2.0*np.pi*i*j/511)*hamming[j]

net.save('net/pretraining.caffemodel')
