import os
import pickle
from pathlib import Path
import trimesh
from trimesh.exchange import obj
import numpy as np
from dataset import sort_vertices_and_faces
from dataset.floorplan_triangles import FPTriangleNodes,FPOriginTriangleNodes
from util.s3d_data_load import read_s3d_mesh_info
from util.visualization import export_mesh_to_shp, plot_vertices_and_faces_with_labels, export_mesh_to_obj
import hydra

from util.misc import scale_vertices, normalize_vertices, shift_vertices

@hydra.main(config_path='../config', config_name='meshgpt', version_base='1.2')
def main(config):
    config.dataset_root = 'G:/workspace_plane2DDL/RefCode/meshgpt-official-version/MeshGPT/data/structure3d'

    dataset = FPTriangleNodes(config, 'train')
    features, target, vertices, faces, face_neighborhood, reverse_op \
        = dataset.get_all_features_for_shape(0)
    # plot_vertices_and_faces_with_labels(vertices, faces, target,'no_reverse.png')
    export_mesh_to_obj(vertices, faces, 'no_reverse.obj')
    for op in reverse_op:
        if op[0] == '+':
            vertices = np.stack([vertices[:, 0] + op[1][0], vertices[:, 1] + op[1][1], vertices[:, 2]], axis=-1)
        elif op[0] == '*':
            vertices = np.stack([vertices[:, 0] * op[1][0], vertices[:, 1] * op[1][1], vertices[:, 2]], axis=-1)
    # plot_vertices_and_faces_with_labels(vertices, faces, target,'reverse.png')
    export_mesh_to_obj(vertices, faces, 'reverse.obj')

    scene_name = dataset.get_name(0)
    scene_full_path = os.path.join(config.dataset_root, scene_name)
    ori_vertices, ori_faces, _, _ = read_s3d_mesh_info(scene_full_path)
    export_mesh_to_obj(ori_vertices, ori_faces, 'ori.obj')

    #dataset.save_sample_data_by_idx(0,True)




if __name__ == '__main__':
    main()