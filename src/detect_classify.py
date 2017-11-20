#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import _init_paths
import caffe, os, sys, cv2, argparse, yaml, librosa, shutil, evalution, time, pdb
import matplotlib.pyplot as plt, soundfile as sf, numpy as np, scipy.io as sio
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

def vis_detections(wav_name, scores, boxes):
    f=open(data_folder+'Result/estimate_txt/'+wav_name[:-4]+'_estimate.txt','wt')
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
		f.write(onset+'\t'+offset+'\t'+name_transform[class_name]+'\n')
    f.close()


if __name__ == '__main__':
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    test_folder='TUT-rare-sound-events-2017-development/data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/audio'

    prototxt = os.path.join('py-R-FCN','models','pascal_voc',args.demo_net,
                            'rfcn_end2end','test_agnostic.prototxt')
    caffemodel_path='py-R-FCN/output/rfcn_end2end_ohem/voc_0712_trainval/'

    labels_plot=['error_rate','deletion_rate','insertion_rate','substitution_rate']
    dictionary_plot={'overall':[], 'baby crying':[], 'glass breaking':[], 'gunshot':[]}

    caffemodel_list=[i for i in os.listdir(caffemodel_path) if i.endswith('.caffemodel')] 
    #caffemodel_list=['py-R-FCN/data/VOCdevkit0712/VOC0712/s15x40_256_9_2017-07-20-17-14-38/resnet50_rfcn_ohem_iter_14000.caffemodel'] 

    name_plot=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    os.makedirs(data_folder+name_plot)
    for i in range(len(caffemodel_list)):
        if(os.path.exists(data_folder+'Result')):
	    shutil.rmtree(data_folder+'Result')
        os.makedirs(data_folder+'Result/estimate_txt')
        #os.makedirs(data_folder+'Result/picture')

    	caffemodel = os.path.join(caffemodel_path,'resnet50_rfcn_ohem_iter_'+str(1000*(i+1))+'.caffemodel')
    	#caffemodel = caffemodel_list[0]

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
	        vis_detections(wav_name, scores, boxes)
        timer.toc()
        print ('Detection took {:.3f}s !').format(timer.total_time)
        print '\nDetect end!\n'
	result=evalution.evalution()
	result_log=open(data_folder+name_plot+'/result.log','at')
	result_log.write(caffemodel+': \n')
	for item_plot in dictionary_plot:
	    list_temp=[]
	    for label_plot in labels_plot:
	        if item_plot=='overall':
		    list_temp.append(result[0]['error_rate'][label_plot])
		    result_log.write(label_plot+': '+str(list_temp[-1])+' \n')
		else:
		    if result[1][item_plot]['error_rate'].has_key(label_plot):
			list_temp.append(result[1][item_plot]['error_rate'][label_plot])
	    dictionary_plot[item_plot].append(list_temp)
	result_log.close()

    if not len(caffemodel_list)==1:
        colors=['r','g','b','y']
        x=np.arange(len(dictionary_plot['overall']))
        for item_plot in dictionary_plot:
	    dictionary_plot[item_plot]=np.array(dictionary_plot[item_plot])
	    np.save(data_folder+name_plot+'/'+item_plot+'.npy',dictionary_plot[item_plot])
    	    fig,axe=plt.subplots(1)
            for i in range(dictionary_plot[item_plot].shape[1]):
	        axe.plot(x,dictionary_plot[item_plot][:,i],color=colors[i],label=labels_plot[i])
            axe.set_title(item_plot)
            axe.legend(loc='best')
	    axe.set_ylabel('rate')
	    axe.set_xlabel('iter/1000')
            axe.set_xlim([0,50])
            axe.set_ylim([0,1.25])
    	    fig.savefig(data_folder+name_plot+'/'+item_plot+'.jpg')
    	    #plt.show()
    	    plt.close()

