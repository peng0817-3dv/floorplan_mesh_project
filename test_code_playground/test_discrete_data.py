import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import openpyxl
import pandas as pd

from dataset import sort_vertices_and_faces_and_labels_and_features
from dataset.floorplan_triangles import read_s3d_mesh_info
from util.misc import normalize_vertices
from util.s3d_data_load import global_label_colors

DATASET_ROOT = 'G:/workspace_plane2DDL/bound_gen_tri_shp'
VISUALIZE_ROOT = 'discrete_data'
CLEAN_RECORD = 'G:/workspace_plane2DDL/bound_gen_tri_shp/clean_record.xlsx'


right_type = ['正确']


def clean_dataset_by_clean_record(record_csv_path):
    data = pd.read_excel(record_csv_path)
    clean_dataset_root = os.path.join( os.path.dirname(DATASET_ROOT),'cleaned_dataset' )
    progress_bar = tqdm(total=len(os.listdir(DATASET_ROOT)))
    if not os.path.exists(clean_dataset_root):
        os.makedirs(clean_dataset_root)

    for scene_name in os.listdir(DATASET_ROOT):
        if not os.path.isdir(os.path.join(DATASET_ROOT, scene_name)):
            continue
        scene_path = os.path.join(DATASET_ROOT, scene_name)
        scene_num = int(scene_name.split('_')[1])
        clean_record = data.loc[scene_num].values
        if clean_record[1] in right_type:
            target_path = os.path.join(clean_dataset_root, scene_name)
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(scene_path, target_path)
        progress_bar.update(1)


def main():
    if not os.path.exists(VISUALIZE_ROOT):
        os.makedirs(VISUALIZE_ROOT)

    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', global_label_colors, N=len(global_label_colors))
    progress_bar = tqdm(total=len(os.listdir(DATASET_ROOT)))
    # count = 1
    for scene_name in os.listdir(DATASET_ROOT):
        if not os.path.isdir(os.path.join(DATASET_ROOT, scene_name)):
            continue
        #
        # count -= 1
        # if count < 0:
        #     break
        scene_path = os.path.join(DATASET_ROOT, scene_name)
        v, f, features, labels = read_s3d_mesh_info(scene_path)
        v = normalize_vertices(v)

        v_d128,f_d128,labels_d128,_ = sort_vertices_and_faces_and_labels_and_features(v,f,labels,features,128)
        v_d256,f_d256,labels_d256,_ = sort_vertices_and_faces_and_labels_and_features(v,f,labels,features,256)
        v_d512,f_d512,labels_d512,_ = sort_vertices_and_faces_and_labels_and_features(v,f,labels,features,512)

        save_path = os.path.join(VISUALIZE_ROOT, f"{scene_name}.png")

        # axs = plt.subplots(2,2)
        # ax1 = axs[0][0]
        # ax2 = axs[0][1]
        # ax3 = axs[1][0]
        # ax4 = axs[1][1]
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

        ax1.add_collection(PolyCollection(\
            np.array([[v[face[0]][:2], v[face[1]][:2], v[face[2]][:2]] for face in f]), \
            array=np.array(labels),\
            cmap=custom_cmap))
        ax1.autoscale()
        ax1.set_title('Original')

        ax2.add_collection(PolyCollection( \
            np.array([[v_d128[face[0]][:2], v_d128[face[1]][:2], v_d128[face[2]][:2]] for face in f_d128]), \
            array=np.array(labels_d128), \
            cmap=custom_cmap))
        ax2.autoscale()
        ax2.set_title('discrete 128')

        ax3.add_collection(PolyCollection( \
            np.array([[v_d256[face[0]][:2], v_d256[face[1]][:2], v_d256[face[2]][:2]] for face in f_d256]), \
            array=np.array(labels_d256), \
            cmap=custom_cmap))
        ax3.autoscale()
        ax3.set_title('discrete 256')

        ax4.add_collection(PolyCollection( \
            np.array([[v_d512[face[0]][:2], v_d512[face[1]][:2], v_d512[face[2]][:2]] for face in f_d512]), \
            array=np.array(labels_d512), \
            cmap=custom_cmap))
        ax4.autoscale()
        ax4.set_title('discrete 512')

        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close("all")
        progress_bar.update(1)





if __name__ == '__main__':
    # main()
    clean_dataset_by_clean_record(CLEAN_RECORD)