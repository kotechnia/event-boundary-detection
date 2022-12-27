import os
import subprocess
from glob import glob
import pandas as pd
from module import *
import parmap


MMACTION_DIR = '../mmaction2/'
PYTHON_ENV = 'python'

CONFIG = {
    'tsn':os.path.join(MMACTION_DIR, 'configs/recognition/tsn/tsn_r50_clip_feature_extraction_1x1x3_rgb.py'),
    'slowfast':os.path.join(MMACTION_DIR,'configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py')
}
CHECKPOINT = {
    'tsn':'https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth',
    'slowfast':'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth',
}




def video_path_to_output_path_dir(video_path):
    """
input:
    video_path
to 
output:
    feature_path
    """

    output_path = video_path.split('/')
    output_path = '/'.join(output_path)

    output_dir = os.path.dirname(output_path)
    output_path, ext = os.path.splitext(output_path)
    
    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs(output_dir, exist_ok = True)
        print(f"Create DIrectory {output_dir}")
        
    return output_path, output_dir


def example(video_path, feature_type='tsn'):
    #video_path = '/mnt/hdd8t/nia2022_1-2_data/videos/일상/220722/D2_DA_0722_000074.mp4'
    
    output_path = None
    
    output_name, output_dir = video_path_to_output_path_dir(video_path)
    print(video_path)
    print(output_name)
    
    temp_video_list_path, temp_video_list = video_segmentation(
        video_path = video_path,
        temp_dir = os.path.join(MMACTION_DIR, 'temp')
    )
    
    try:
        output_path = feature_extract(
            video_list_path = temp_video_list_path,
            output_name = output_name,
            feature_type = feature_type 
        )
    except (subprocess.CalledProcessError) as e:
        print(e)
        print(output_name)

    _ = delete_temporary_file(temp_video_list_path)
    return output_path

def process_function(params):
    value = params
    "----------------------------------------------------------------------"
    
    video_path = value['video_path']
    #feature_path = example(video_path, feature_type='slowfast')
    feature_path = example(video_path, feature_type='tsn')
    value['feature_path'] = feature_path
    
    "----------------------------------------------------------------------"
    return value


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--core', default=10)
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    def get_key(path):
        key = os.path.basename(path)
        key, _ = os.path.splitext(key)
        key = key.split('_')
        if key[-1] in ['tsn', 'slowfast']:
	        key = key[:-1]
        key = '_'.join(key)
        return key

    videos = glob('../data/**/*.mp4', recursive=True)
    features = glob('../data/**/*_tsn.pkl', recursive=True)
	
    videos = pd.DataFrame({'video_path':videos})
    videos['key'] = videos['video_path'].map(get_key)
    features = pd.DataFrame({'feature_path':features})
    features['key'] = features['feature_path'].map(get_key)
    videos = pd.merge(videos, features, on='key', how='outer')
    videos = videos[videos['feature_path'].isnull()].reset_index(drop=True)
    
    process_params = [videos.loc[i].to_dict() for i in reversed(range(len(videos)))]
    result = parmap.map(process_function, process_params, pm_pbar=True, pm_processes=int(args.core))
    print(pd.DataFrame(result))

