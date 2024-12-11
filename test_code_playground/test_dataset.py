import os
import pickle
from pathlib import Path
import trimesh
from trimesh.exchange import obj
import numpy as np
from dataset import sort_vertices_and_faces
from dataset.floorplan_triangles import FPTriangleNodes,FPOriginTriangleNodes
from util.visualization import export_mesh_to_shp
import hydra

from util.misc import scale_vertices, normalize_vertices, shift_vertices

@hydra.main(config_path='../config', config_name='meshgpt', version_base='1.2')
def main(config):
    config.dataset_root = 'G:/workspace_plane2DDL/RefCode/meshgpt-official-version/MeshGPT/data/structure3d'

    dataset = FPTriangleNodes(config, 'train')
    data = dataset.get(0)
    print(data.x.shape)

    #dataset.save_sample_data_by_idx(0,True)




if __name__ == '__main__':
    main()