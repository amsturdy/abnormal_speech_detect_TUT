import numpy as np
import cv2
import os

def get_mean(folder):
    image_mean=np.zeros((1,1,3))
    count=0
    height=[]
    for file in sorted(os.listdir(folder+'JPEGImages')):
	if(file.endswith('.jpg')):
	    count+=1
            full_name=os.path.join(folder,'JPEGImages',file)
	    im=cv2.imread(full_name)
	    if((int(file[4:-4])-1)%8==0):
		height.append(im.shape[0])
	    for i in range(3):
	        image_mean[0,0,i] += np.mean(im[:,:,i])
    height=sorted(height)
    print(len(height),height[0],height[-1])
    image_mean /= (1.0*count)
    np.save(folder+'image_mean.npy', image_mean)
    return image_mean

if __name__=='__main__':
    folder='py-R-FCN/data/VOCdevkit0712/VOC0712/'
    if(os.path.exists(folder+'/image_mean.npy')):
	print(np.load(folder+'image_mean.npy'))
    else:
        print(get_mean(folder))
