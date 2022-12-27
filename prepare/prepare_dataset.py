import pandas as pd
import json
import os
from glob import glob

def get_key(path):
    path = os.path.basename(path)
    key, _ = os.path.splitext(path)
    return key

def main(args):

	df_json_list = pd.DataFrame({'json_path':json_list})
	df_json_list['key'] = df_json_list['json_path'].map(get_key)

	try:
    	df.insert(df.shape[1], 'error', None)
	except:
    	pass

	all_data = {}
	for i in tqdm(range(len(df))):
	    try:
        	json_path = df.loc[i, 'json_path']
        	with open(json_path) as f:
            	data = json.load(f)

        	trigger_info = data.pop('trigger_info')
        	for j in range(len(trigger_info)):
            	timestamps = trigger_info[j].pop('timestamps')
            	trigger_info[j]['start'] = get_time(timestamps[0])
            	try:
                	trigger_info[j]['end'] = get_time(timestamps[1])
            	except (IndexError) as e:
                	trigger_info[j]['end'] = get_time(timestamps[0])

        	local_df = pd.DataFrame(trigger_info)
        	change_cuts = local_df.loc[local_df['trigger'] == "Change due to cut"]
        	change_cuts = np.unique(change_cuts[['start', 'end']]).tolist()
        	change_cuts = [x for x in change_cuts if x not in [0., data['duration']] ]
        
        	change_acts = local_df.loc[local_df['trigger'] == "Change of action"]
        	change_acts = np.unique(change_acts[['start', 'end']]).tolist()
        	change_acts = [x for x in change_acts if x not in [0., data['duration']] ]

        	change_events = change_acts + change_cuts

        	key = get_key(data['video_name'])
        	fps = float(data['frame_rate'])

        	all_data[key]={
            	'path_video':data['video_name'],
            	'path_frame':data['video_name'],
            	'fps':fps,
            	'duration':float(data['duration']),
            	'num_frames':int(data['total_frame']),
            	'f1_consis':data['f1_consis'],
            	'f1_consis_avg':data['f1_consis_avg'],
            	'change_event':[change_acts],
            	'change_shot':[change_cuts],
            	'substages_myframeidx':[list(map(lambda x : x*fps, change_events))],
            	'substages_timestamps':[change_events],
        	}
    	except Exception as e:
        	df.loc[i, 'error'] = e

	with open("all_data.json", "w", encoding='utf-8-sig') as f:
    	json.dump(all_data, f, indent=4)

	if True:
		temp_df = df[~df['error'].isnull()].reset_index(drop=True)
		for i in range(len(temp_df)):
    		print(f"{temp_df.loc[i, 'json_path']}, {temp_df.loc[i, 'error']}")

if __name__ == '__main__':
	args = []
	main(args)
