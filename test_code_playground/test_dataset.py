import os
import pickle
from pathlib import Path
import trimesh
from trimesh.exchange import obj
import numpy as np

from ablation.only_segment_room_and_wall import FPTriangleWithThreeClsNodes
from dataset import sort_vertices_and_faces
from dataset.floorplan_triangles import FPTriangleNodes,FPOriginTriangleNodes
from util.s3d_data_load import read_s3d_mesh_info
from util.visualization import export_mesh_to_shp, plot_vertices_and_faces_with_labels, export_mesh_to_obj
import hydra

from util.misc import scale_vertices, normalize_vertices, shift_vertices

@hydra.main(config_path='../config', config_name='only_segment_room_and_wall', version_base='1.2')
def main(config):
    config.dataset_root = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_featured_shp" # 测试数据集位置
    config.train_ratio = 1.0
    config.val_ratio = 0.0
    config.test_ratio = 0.0
    dataset = FPTriangleWithThreeClsNodes(config,'train',False,False)
    print(len(dataset))
    #dataset.save_sample_data_by_idx(0,True)




if __name__ == '__main__':
    main()