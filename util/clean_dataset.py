import os
import shutil
import pandas as pd
from tqdm import tqdm

right_type = ['正确']

def clean_dataset_by_clean_record(dataset_root,record_csv_path):
    data = pd.read_excel(record_csv_path)
    clean_dataset_root = os.path.join( os.path.dirname(dataset_root),'cleaned_dataset' )
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

if __name__ == '__main__':
    CLEAN_RECORD = 'G:/workspace_plane2DDL/bound_gen_tri_shp/clean_record.xlsx'
    DATASET_ROOT = r'G:\workspace_plane2DDL\testData\floorplane_result_shp'
    clean_dataset_by_clean_record(DATASET_ROOT,CLEAN_RECORD)