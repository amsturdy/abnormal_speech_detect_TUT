#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import _init_paths
import caffe, os, sys, cv2, argparse, yaml, librosa, shutil, pdb
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import scipy.io as sio
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from PIL import Image
from mystfft import stfft

data_folder='py-R-FCN/data/VOCdevkit0712/VOC0712/'
#CONF_THRESH = [1,0.6,0.6,0.6]
CONF_THRESH = [1,0.37,0.5,0.51]
NMS_THRESH = 0.3
common_fs=44100
nfft=1024
noverlap=nfft-441

CLASSES = ('__background__',
           'babycry', 'glassbreak', 'gunshot')

name_transform={'babycry':'baby crying','glassbreak':'glass breaking','gunshot':'gunshot'}

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}

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

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ResNet-101]',
                        default='ResNet-101')

    args = parser.parse_args()

    return args

def vis_detections(im, wav_name, scores, boxes):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    im_size=im.shape
    fig, ax = plt.subplots(figsize=(im_size[1]/100.0, im_size[0]/100.0))
    ax.imshow(im, aspect='equal')
    result=open('task2_results.txt','at')
    result.write('audio/'+wav_name)
    write_result=False
    possible_class=np.argmax(scores,axis=1)
    possible_dets={1:[],2:[],3:[]}
    for i,Class in enumerate(possible_class):
	if not Class==0:
	    item = [0,boxes[i,2],511,boxes[i,3],scores[i][Class]]
	    possible_dets[Class].append(item)

    for Class in possible_dets:
	dets= np.array(possible_dets[Class],dtype=np.float32)
	if len(dets)==0:
	    continue
	class_name = CLASSES[Class]
        keep = nms(dets, NMS_THRESH)
	dets = dets[keep, :]
	if len(dets) == 0:
	    continue
	else:
	    for i in range(dets.shape[0]):
		onset=str(dets[i,1]*(nfft-noverlap)/common_fs)
		offset=str(dets[i,3]*(nfft-noverlap)/common_fs)
		if not write_result:
		    result.write('\t'+onset+'\t'+offset+'\t'+class_name)
		    write_result=True
		else:
		    result.write('\n'+'audio/'+wav_name+'\t'+onset+'\t'+offset+'\t'+class_name)
                ax.add_patch(
                    plt.Rectangle((dets[i,0], dets[i,1]),
                          dets[i,2] - dets[i,0],
                          dets[i,3] - dets[i,1], fill=False,
                          edgecolor='red', linewidth=3.5)
                )
                ax.text(dets[i,0], dets[i,1] - 2,
                    '{:s} {:.3f}'.format(class_name, dets[i,-1]),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
    result.write('\n')
    result.close()
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.show(block=False)
    #plt.pause(0.1)
    save_image_name=os.path.join(data_folder,'Result/picture',wav_name[:-4]+'.jpg')
    plt.savefig(save_image_name)
    plt.close()

if __name__ == '__main__':
    if(os.path.exists('task2_results.txt')):
	os.remove('task2_results.txt')
    
    if(os.path.exists(data_folder+'Result')):
	shutil.rmtree(data_folder+'Result')
    os.makedirs(data_folder+'Result/picture')

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    test_folder='TUT-rare-sound-events-2017-evaluation/audio'

    prototxt = os.path.join('py-R-FCN','models','pascal_voc',args.demo_net,
                            'rfcn_end2end', 'test_agnostic.prototxt')
    #caffemodel_path='py-R-FCN/output/rfcn_end2end_ohem/voc_0712_trainval/'
    caffemodel_path='py-R-FCN/data/VOCdevkit0712/VOC0712/s15x40_256_9_2017-07-20-17-14-38/'
    caffemodel = os.path.join(caffemodel_path,'resnet50_rfcn_ohem_iter_14000.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    #for file in sorted(os.listdir(caffemodel_path)):
	#if file.endswith('.caffemodel'):
    	   # caffemodel = os.path.join(caffemodel_path,file)
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
      	caffe.set_mode_gpu()
       	caffe.set_device(args.gpu_id)
       	cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((1000, 512, 3), dtype=np.uint8)
    for i in xrange(2):
        _, __= im_detect(net, im)
   
    print '\nDetect begin: \n'
    timer=Timer()
    timer.tic()
    for wav_name in sorted(os.listdir(test_folder)):
	if wav_name.endswith('.wav'):
	    full_wav_name=os.path.join(test_folder,wav_name)		
            x,rate=load_audio(full_wav_name,common_fs)
	    image=stfft(x,nfft,noverlap)
	    IMG=Image.fromarray(image.astype(np.uint8))
	    IMG.save('temp.jpg')
	    im=cv2.imread('temp.jpg')
	    os.remove('temp.jpg')
	    '''
	    im=np.zeros((image.shape[0],image.shape[1],3))
	    for k in range(3):
	        im[:,:,k]=np.uint8(image)
	    '''
	    #timer.tic()
	    scores, boxes = im_detect(net, im)
	    #timer.toc()   
	    #print ('Detection took {:.3f}s for '
		        #'{:d} object proposals').format(timer.total_time, boxes.shape[0])

	    # Visualize detections for each class
	    vis_detections(im, wav_name, scores, boxes)
    timer.toc()
    print ('Detection took {:.3f}s !').format(timer.total_time)
    print '\nDetect end!\n'
