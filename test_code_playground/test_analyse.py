import os

import numpy as np
from tqdm import tqdm

from util.analyse_dataset import random_split, analyse_dataset_split
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
if __name__ == '__main__':
    main()