import os
import pickle
from pathlib import Path
import trimesh
from trimesh.exchange import obj
import numpy as np
from dataset import sort_vertices_and_faces
from dataset.floorplan_triangles import FPTriangleNodes
import hydra

from util.misc import scale_vertices, normalize_vertices, shift_vertices

@hydra.main(config_path='../config', config_name='meshgpt', version_base='1.2')
def main(config):
    sample_size = 32

    config.dataset_root = 'G:/workspace_plane2DDL/RefCode/meshgpt-official-version/MeshGPT/data/structure3d'
    augmentation_data_root = 'G:/workspace_plane2DDL/RefCode/meshgpt-official-version/MeshGPT/test_code_playground/autmentation_data'
    if not os.path.exists(augmentation_data_root):
        os.makedirs(augmentation_data_root)
    dataset = FPTriangleNodes(config, 'train')
    for id ,data in enumerate(dataset):
        print(id)
    #dataset.save_sample_data_by_idx(0,True)




if __name__ == '__main__':
    main()