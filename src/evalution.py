import sed_eval,yaml,os,shutil

def read_meta_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.load(infile)
    return data

def prepare_result_txt():
    folder='TUT-rare-sound-events-2017-development/data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/'
    result_folder='py-R-FCN/data/VOCdevkit0712/VOC0712/Result/'
    if(os.path.exists(result_folder+'reference_txt')):
        shutil.rmtree(result_folder+'reference_txt')
    os.makedirs(result_folder+'reference_txt')

    name_transform={'babycry':'baby crying','glassbreak':'glass breaking','gunshot':'gunshot'}
    event_labels=['baby crying', 'glass breaking', 'gunshot']
    babycry_yaml=folder+'meta/mixture_recipes_devtest_babycry.yaml'
    glassbreak_yaml=folder+'meta/mixture_recipes_devtest_glassbreak.yaml'
    gunshot_yaml=folder+'meta/mixture_recipes_devtest_gunshot.yaml'
    file_list={'all':[],'baby crying':[],'glass breaking':[],'gunshot':[],'-6.0':[],'0.0':[],'6.0':[],'-inf':[]}
    for i,class_yaml in enumerate([babycry_yaml,glassbreak_yaml,gunshot_yaml]):
        data=read_meta_yaml(class_yaml)
        for item in data:
            base_name=os.path.splitext(item['mixture_audio_filename'])[0]
	    if not os.path.exists(folder+'audio/'+base_name+'.wav'):
		continue
	    if item['event_present']:
                reference_name=base_name+('_'+str(item['ebr'])+'_'+item['event_class'])
	        f=open(result_folder+'reference_txt/'+reference_name+'_reference.txt','wt')
	        f.write(str(item['event_start_in_mixture_seconds'])+'\t'+
			str(item['event_start_in_mixture_seconds']+item['event_length_seconds'])+'\t'+
			name_transform[item['event_class']]
		       )
	        f.close()
	    else:
	        reference_name=base_name
	        f=open(result_folder+'reference_txt/'+reference_name+'_reference.txt','wt')
	        f.close
	    file_list[event_labels[i]].append({
					       'reference_file': result_folder+'reference_txt/'+reference_name+'_reference.txt',
					       'estimated_file': result_folder+'estimate_txt/'+base_name+'_estimate.txt'
					      })
	    file_list[str(item['ebr'])].append({
					        'reference_file': result_folder+'reference_txt/'+reference_name+'_reference.txt',
					        'estimated_file': result_folder+'estimate_txt/'+base_name+'_estimate.txt'
					       })
	    file_list['all'].append({
				     'reference_file': result_folder+'reference_txt/'+reference_name+'_reference.txt',
				     'estimated_file': result_folder+'estimate_txt/'+base_name+'_estimate.txt'
			           })
    return file_list

def evalution():
    event_labels=['baby crying', 'glass breaking', 'gunshot']
    data = []

    # Get used event labels
    all_data = sed_eval.util.event_list.EventList()
    file_list=prepare_result_txt()
    for file_pair in file_list['all']: #+file_list['0.0']+file_list['-6.0']:
        reference_event_list = sed_eval.io.load_event_list(file_pair['reference_file'])
        estimated_event_list = sed_eval.io.load_event_list(file_pair['estimated_file'])
        data.append({'reference_event_list': reference_event_list,
                     'estimated_event_list': estimated_event_list})
        all_data += reference_event_list
    #event_labels = all_data.unique_event_labels

    # Start evaluating

    # Create metrics classes, define parameters
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=event_labels,
                                                                 time_resolution=1)
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(event_label_list=event_labels,
							     evaluate_onset=True,
							     evaluate_offset=False,
                                                             t_collar=0.5)

    # Go through files
    for file_pair in data:
        segment_based_metrics.evaluate(file_pair['reference_event_list'],
                                   file_pair['estimated_event_list'])
        event_based_metrics.evaluate(file_pair['reference_event_list'],
                                 file_pair['estimated_event_list'])

    # Get only certain metrics
    overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
    print "Accuracy:", overall_segment_based_metrics['accuracy']['accuracy']

    # Or print all metrics as reports
    print segment_based_metrics
    print event_based_metrics
    overall=event_based_metrics.results_overall_metrics()
    class_wise=event_based_metrics.results_class_wise_metrics()

    result=[overall,class_wise]
    return result

if __name__ == '__main__':
    print evalution()[0]['error_rate']['error_rate']
