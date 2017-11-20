#!/bin/bash

:<<not_run

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~generate data start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cd TUT-rare-sound-events-2017-development/TUT_Rare_sound_events_mixture_synthesizer
python generate_devtest_mixtures.py 
if [ $? != 0 ];then
    echo "generate_devtest_mixtures.py execute failed!"
    exit 1
fi

python generate_devtrain_mixtures.py
if [ $? != 0 ];then
    echo "generate_devtrain_mixtures.py execute failed!"
    exit 1
fi
cd ../..
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~extract feature start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python ./src/extract_feature.py
if [ $? != 0 ];then
    echo "./src/extract_feature.py execute failed!"
    exit 1
fi
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~make dataset start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python ./src/write_xml.py
if [ $? != 0 ];then
    echo "./src/write_xml.py execute failed!"
    exit 1
fi
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
not_run


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~train start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cd py-R-FCN
./experiments/scripts/rfcn_end2end_ohem.sh 0 ResNet-50 pascal_voc
#./experiments/scripts/rfcn_end2end_ohem.sh 0 MobileNet pascal_voc
#./experiments/scripts/rfcn_end2end_ohem.sh 0 VGG16 pascal_voc
#./experiments/scripts/rfcn_end2end_ohem.sh 0 ZF pascal_voc
#./experiments/scripts/rfcn_alt_opt_5stage_ohem.sh 0 ResNet-50 pascal_voc
if [ $? != 0 ];then
    echo "./py-R-FCN/experiments/scripts/rfcn_end2end_ohem.sh execute failed!"
    exit 1
fi
cd ..
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~detect start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
python ./src/detect_classify.py --net ResNet-50
#python ./src/detect_classify.py --net MobileNet 
#python ./src/detect_classify.py --net VGG16
#python ./src/detect_classify.py --net ZF
if [ $? != 0 ];then
    echo "./src/detect_classify.py execute failed!"
    exit 1
fi
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~evalution start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#python ./src/evalution.py
#if [ $? != 0 ];then
#    echo "./src/evalution.py execute failed!"
#    exit 1
#fi
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
