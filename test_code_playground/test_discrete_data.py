import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from dataset import sort_vertices_and_faces_and_labels_and_features
from dataset.floorplan_triangles import read_vertexes_and_faces
from util.misc import normalize_vertices

DATASET_ROOT = 'G:/workspace_plane2DDL/bound_gen_tri_shp'
VISUALIZE_ROOT = 'discrete_data'

colors = [
'#e7c9b7',
'#5c6bc0',
'#ff5733',
'#1e88e5',
'#cddc39',
'#f06292',
'#ffa726',
'#9c27b0',
'#81c784',
'#64b5f6',
'#ffb74d',
'#90caf9',
'#78909c',
'#ffecb3',
'#a1887f',
'#d32f2f',
'#1976d2',
'#388e3c',
'#7e57c2',
'#fb8c00',
'#f44336',
'#26a69a',
'#f48fb1',
'#ffb300',
'#e57373',
'#64dd17',
'#ffcc80',
'#8e24aa',
'#7b1fa2',
'#e1bee7',
'white',
'black'
]

label_names = [
'living_room' ,
'kitchen',
'bedroom',
'bathroom',
'balcony',
'corridor',
'dining_room',
'study',
'studio',
'store_room',
'garden',
'laundry_room',
'office',
'basement',
'garage',
'undefined',
'door',
'window',
'out_wall',
'in_wall'
]




def main():
    if not os.path.exists(VISUALIZE_ROOT):
        os.makedirs(VISUALIZE_ROOT)

    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
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
        v, f, features, labels = read_vertexes_and_faces(scene_path)
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
    main()