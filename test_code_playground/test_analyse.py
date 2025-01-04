import os
import shutil

import numpy as np
from tqdm import tqdm

from util.analyse_dataset import random_split, analyse_dataset_split, shp_to_obj
from util.s3d_data_load import read_s3d_mesh_info



def main():
    dataset_root = "G:/workspace_plane2DDL/bound_gen_tri_shp"
    label_datas = []
    n = 0
    # Read all the labels of the dataset
    progress_bar = tqdm(os.listdir(dataset_root))
    for scene in os.listdir(dataset_root):
        progress_bar.update(1)
        if not os.path.isdir(os.path.join(dataset_root, scene)):
            continue
        mesh_info_path = os.path.join(dataset_root, scene)
        _ ,_ ,_ ,label = read_s3d_mesh_info(mesh_info_path)
        label_datas.append(label)
        n += 1
    print("Total number of scenes: ", n)
    sequence_idx = np.arange(n)
    sequence_split = [sequence_idx[0:int(n * 0.7)] ,sequence_idx[int(n * 0.7):int(n * 0.9)] ,sequence_idx[int(n * 0.9):]]
    rd_split = random_split(n, 0.7, 0.2)
    analyse_dataset_split(rd_split, label_datas, analyse_results_path = "random_split_3500.png")
    print("analyse dataset random_split done")
    analyse_dataset_split(sequence_split, label_datas, analyse_results_path = "sequence_split_3500.png")
    print("analyse dataset sequence_split done")


def test_shp_to_obj():
    shp_root = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_anno"

    tq_bar = tqdm(os.listdir(shp_root))
    for scene in os.listdir(shp_root):
        shp_path = os.path.join(shp_root, scene, "GT_room_poly.shp")
        obj_path = os.path.join(shp_root, scene, f"{scene}.obj")
        shp_to_obj(shp_path, obj_path)
        tq_bar.update(1)


def mov():
    obj_root = r"G:\workspace_plane2DDL\real_point_cloud_dataset\scaled_stru3d_anno"
    las_root = r"C:\Users\Peng\Downloads\0-400"
    tq_bar = tqdm(os.listdir(las_root))
    copy_shp_file = ["GT_room_poly.dbf","GT_room_poly.shp","GT_room_poly.shx"]
    for scene in os.listdir(las_root):
        tq_bar.update(1)
        # target_path = os.path.join(las_root, scene,f"{scene}.obj")
        # src_path = os.path.join(obj_root, scene, f"{scene}.obj")
        # if not os.path.exists(src_path):
        #     continue
        # shutil.copyfile(src_path, target_path)
        for file in copy_shp_file:
            src_file = os.path.join(obj_root, scene, file)
            target_file = os.path.join(las_root, scene, file)
            if os.path.exists(src_file):
                shutil.copyfile(src_file, target_file)



if __name__ == '__main__':
    # main()
    test_shp_to_obj()
    # mov()
