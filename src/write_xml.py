import glob
import pdb
import os
import shutil

s1="""    <object>
        <name>{0}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{1}</xmin>
            <ymin>{2}</ymin>
            <xmax>{3}</xmax>
            <ymax>{4}</ymax>
        </bndbox>
    </object>"""

s2="""<annotation>
    <folder>VOC2007</folder>
    <filename>{0}</filename>
    <source>
        <database>My Database</database>
        <annotation>VOC2007</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>wangkaiwu</name>
    </owner>
    <size>
        <width>{1}</width>
        <height>{2}</height>
        <depth>{3}</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>{4}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{5}</xmin>
            <ymin>{6}</ymin>
            <xmax>{7}</xmax>
            <ymax>{8}</ymax>
        </bndbox>
    </object>{9}
</annotation>
"""

data='py-R-FCN/data/VOCdevkit0712/VOC0712/'
if(os.path.exists(data+'Annotations')):
    shutil.rmtree(data+'Annotations')
os.makedirs(data+'Annotations')
textlist=glob.glob(data+'Labels/*.txt')
for text_ in textlist:
    flabel = open(text_, 'r')
    lb = flabel.readlines()
    flabel.close()
    ob2 = ""
    if len(lb)<4:
        continue  # no annotation
    l1=lb[1].split(' ')
    l2=lb[3].split(' ')
    x1=[int(i) for i in l1]
    #pdb.set_trace()
    x2=[int(i) for i in l2]
    #pdb.set_trace()
    if len(lb)>4:  # extra annotation
        for i in range(4,len(lb),2):
            l = lb[i+1].split(' ')
	    y = [int(j) for j in l]
	    #pdb.set_trace()
            ob2+='\n' + s1.format(lb[i][0:-1],y[0],y[1],y[2],y[3])
	#pdb.set_trace()
    imgname=lb[0][0:-1]+'.jpg'
    savename=data+'Annotations/'+lb[0][0:-1]+'.xml'
    f = open(savename, 'w')
    ob1=s2.format(imgname, x1[0],x1[1],x1[2], lb[2][0:-1], x2[0],x2[1],x2[2],x2[3],  ob2)
    f.write(ob1)
    f.close()
