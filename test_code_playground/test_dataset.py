import os
import pickle
import shutil
from pathlib import Path
import trimesh
from trimesh.exchange import obj
import numpy as np

from ablation.only_segment_room_and_wall import FPTriangleWithThreeClsNodes
from dataset import sort_vertices_and_faces
from dataset.floorplan_triangles import FPTriangleNodes,FPOriginTriangleNodes
from util.s3d_data_load import read_s3d_mesh_info
from util.s3d_data_process import process_vertice_by_op_record
from util.visualization import export_mesh_to_shp, plot_vertices_and_faces_with_labels, export_mesh_to_obj
import hydra

from util.misc import scale_vertices, normalize_vertices, shift_vertices

@hydra.main(config_path='../config', config_name='only_segment_room_and_wall', version_base='1.2')
def main(config):
    config.dataset_root = r"G:/workspace_plane2DDL/testData\anno_shp_with_normal_diffusion" # 测试数据集位置

    dataset = FPTriangleWithThreeClsNodes(config,'test',split_mode="scene_names")

    n = len(dataset)
    print(f"dataset len:{n}")
    print(f"start:{dataset.get_name(0)}")
    print(f"end:{dataset.get_name(n-1)}")
    data1 = dataset[0]
    print("done")

if __name__ == '__main__':
    main()