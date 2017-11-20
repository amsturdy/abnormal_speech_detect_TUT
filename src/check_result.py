import os, shutil, numpy as np

if __name__ == '__main__':
    result_txt='task2_results.txt'
    if os.path.exists('check_result'):
	shutil.rmtree('check_result')
    os.makedirs('check_result')

    result=open(result_txt,'r')
    lines=result.readlines()
    result.close()

    f=open('check_result.txt','w')
    folder='TUT-rare-sound-events-2017-evaluation'
    shuffle=np.random.permutation(len(lines))
    for i in range(100):
	line=lines[shuffle[i]]
	name=line.split('\t')[0]
	audio=os.path.join(folder,name)
	if not os.path.exists(audio):
	    audio=audio[:-1]
	f.write(line+'\n\n')
	shutil.copy(audio,'check_result')
    f.close()
