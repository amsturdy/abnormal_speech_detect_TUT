# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

import _init_paths
import stfft,yaml,librosa,argparse,cv2,caffe,cPickle,os,sys,shutil,pdb

import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import soundfile as sf
import scipy.io as sio

from PIL import Image
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms
from utils.blob import im_list_to_blob

data_folder='py-R-FCN/data/VOCdevkit0712/VOC0712/'
CONF_THRESH = [1,0.65,0.5,0.4]
NMS_THRESH = 0.3
common_fs=44100
nfft=1023
noverlap=512

CLASSES = ('__background__',
           'babycry', 'glassbreak', 'gunshot')

name_transform={'babycry':'baby crying','glassbreak':'glass breaking','gunshot':'gunshot'}

NETS = {'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel')}
  
def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    #image_mean='py-R-FCN/data/VOCdevkit0712/VOC0712/image_mean.npy'
    #assert os.path.exists(image_mean), "please run src/compute_mean.py first!"
    #PIXEL_MEANS=np.load(image_mean)
    #im_orig -= PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = [im_orig]
    im_scale_factors = [1]

    #processed_ims = []
    #im_scale_factors = []

    #for target_size in cfg.TEST.SCALES:
    #    im_scale = float(target_size) / float(im_size_min)
    #    # Prevent the biggest axis from being more than MAX_SIZE
    #    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
    #        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    #    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
    #                    interpolation=cv2.INTER_LINEAR)
    #    im_scale_factors.append(im_scale)
    #    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
    
    features=net.blobs['res4f'].data

    return features[0]

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=400, thresh=-np.inf, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            if cfg.TEST.AGNOSTIC:
                cls_boxes_y = boxes[inds, 2:]
            else:
                cls_boxes = boxes[inds, j*4:(j+1)*4]
	    cls_boxes = np.zeros((cls_boxes_y.shape[0],4),cls_boxes_y.dtype)
	    cls_boxes[:,1] = 255
	    cls_boxes[:,2] = cls_boxes_y[:,0]
	    cls_boxes[:,3] = cls_boxes_y[:,1]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)

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
    '''
    im = im[:, :, (2, 1, 0)]
    im_size=im.shape
    fig, ax = plt.subplots(figsize=(im_size[1]/100.0, im_size[0]/100.0))
    ax.imshow(im, aspect='equal')
    '''
    result=open(data_folder+'Result/task2_results.txt','at')
    f=open(data_folder+'Result/estimate_txt/'+wav_name[:-4]+'_estimate.txt','wt')
    result.write(wav_name)
    write_result=False
    cls_boxes = np.zeros((boxes.shape[0],4),boxes.dtype)
    cls_boxes[:,1] = boxes[:,2]
    cls_boxes[:,2] = 511
    cls_boxes[:,3] = boxes[:,3]
    for cls_ind, class_name in enumerate(CLASSES[1:]):
        cls_ind += 1# because we skipped background
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        #np.save('data/'+im_name+'_keep.npy',keep)
        #sio.savemat('data/'+im_name+'_keep.mat',{'keep':keep})
        #dets = dets[keep, :]

	#dets=dets[np.argmax(dets[:,-1]),:]
        #dets=dets[np.newaxis,:]
	
	'''
	for i in keep:
	    if dets[i,-1]>(CONF_THRESH[cls_ind]+0.5*(1-scores[i,0])):

	'''
	dets = dets[keep, :]
	inds = np.where(dets[:, -1] > CONF_THRESH[cls_ind])[0]
	if len(inds) == 0:
	    continue
	else:
	    for i in inds:
		onset=str(dets[i,1]*(nfft-noverlap)/common_fs)
		offset=str(dets[i,3]*(nfft-noverlap)/common_fs)
		if not write_result:
		    result.write('\t'+onset+'\t'+offset+'\t'+name_transform[class_name])
		    write_result=True
		else:
		    result.write('\n'+wav_name+'\t'+onset+'\t'+offset+'\t'+name_transform[class_name])
		f.write(onset+'\t'+offset+'\t'+name_transform[class_name]+'\n')
		'''
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
		'''
    result.write('\n')
    result.close()
    f.close()
    '''
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.show(block=False)
    #plt.pause(0.1)
    save_image_name=os.path.join(data_folder,'Result/picture',wav_name[:-4]+'.jpg')
    plt.savefig(save_image_name)
    plt.close()
    '''

if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join('py-R-FCN','models','pascal_voc',args.demo_net,
                            'rfcn_end2end', 'test_agnostic.prototxt')
    caffemodel = os.path.join('py-R-FCN','output', 'rfcn_end2end_ohem','voc_0712_trainval',
                              'resnet50_rfcn_ohem_iter_20000.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    folder='TUT-rare-sound-events-2017-development/data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/audio'

    # Warmup on a dummy image
    im = 128 * np.ones((1000, 512, 3), dtype=np.uint8)
    for i in xrange(2):
        _ = im_detect(net, im)

    print '\nDetect begin: \n'
    for wav_name in os.listdir(folder):
        if wav_name.endswith('.wav'):
            full_wav_name=os.path.join(folder,wav_name)
            x,rate=load_audio(full_wav_name,common_fs)
            image=stfft.stfft(x,nfft,noverlap)
            im=np.zeros((image.shape[0],image.shape[1],3))
            for k in range(3):
                im[:,:,k]=np.uint8(image)
            features = im_detect(net, im)
	    for p in range(features.shape[0]):
		temp=features[p]
                IMG=Image.fromarray(temp.astype(np.uint8))
                IMG.save('no_zuo_no_die/'+wav_name+'_'+str(p)+'_.jpg')
    print '\nDetect end: \n'

