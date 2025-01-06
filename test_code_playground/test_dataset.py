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
    config.dataset_root = r"G:\workspace_plane2DDL\testData\test_sample" # 测试数据集位置
    config.train_ratio = 0.0
    config.val_ratio = 0.0
    config.test_ratio = 1.0
    dataset = FPTriangleWithThreeClsNodes(config,'test',True,True)

    test_scene_id = 0

    _, labels, vertices, faces, _, op = dataset.get_all_features_for_shape(test_scene_id)
    vertices = process_vertice_by_op_record(op, vertices)
    output_path = f"test_scene_{test_scene_id}"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    export_mesh_to_shp(vertices,faces,labels,output_path)


if __name__ == '__main__':
    main()