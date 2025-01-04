import os
import shutil
import pandas as pd
from tqdm import tqdm

right_type = ['正确']

def clean_dataset_by_clean_record(dataset_root,record_csv_path):
    data = pd.read_excel(record_csv_path)
    data_name = os.path.basename(dataset_root)
    clean_dataset_root = os.path.join( os.path.dirname(dataset_root),f'{data_name}_cleaned')
    progress_bar = tqdm(total=len(os.listdir(dataset_root)))
    if not os.path.exists(clean_dataset_root):
        os.makedirs(clean_dataset_root)

    for scene_name in os.listdir(dataset_root):
        if not os.path.isdir(os.path.join(dataset_root, scene_name)):
            continue
        scene_path = os.path.join(dataset_root, scene_name)
        scene_num = int(scene_name.split('_')[1])
        clean_record = data.loc[scene_num].values
        if clean_record[1] in right_type:
            target_path = os.path.join(clean_dataset_root, scene_name)
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(scene_path, target_path)
        progress_bar.update(1)

    return clean_dataset_root
