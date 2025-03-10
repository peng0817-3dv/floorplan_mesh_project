import json
import os
import pickle
import shutil
from pathlib import Path

import cv2
import trimesh
from trimesh.exchange import obj
import numpy as np
import time
from ablation.only_segment_room_and_wall import FPTriangleWithThreeClsNodes
from dataset import sort_vertices_and_faces
from dataset.floorplan_triangles import FPTriangleNodes,FPOriginTriangleNodes
from srcipt.generate_coco_stru3d import parse_coco_dict
from util.Evaluator import Evaluator
from util.merge_polygon import MergePolygonSolution
from util.s3d_data_load import read_s3d_mesh_info
from util.s3d_data_process import process_vertice_by_op_record, process_vertice_by_ori_bound
from util.visualization import plot_trimesh_with_labels, plot_floorplan_with_rooms_and_bound
import hydra
from dataset.floorplan_triangles import FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes
from util.misc import scale_vertices, normalize_vertices, shift_vertices

@hydra.main(config_path='../config', config_name='graph_transformer', version_base='1.2')
def main(config):
    config.dataset_root = r"G:/workspace_plane2DDL/real_point_cloud_dataset\test_stru3d_featured_shp" # 测试数据集位置

    dataset = FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes(config,'test',split_mode="scene_names")

    n = len(dataset)
    print(f"dataset len:{n}")
    print(f"start:{dataset.get_name(0)}")
    print(f"end:{dataset.get_name(n-1)}")
    data1 = dataset[0]

    _, targets, vertices, faces, _, ori_bound = dataset.get_all_features_for_shape(0)
    scene_name = dataset.get_name(0)
    scene_num = int(scene_name.split("_")[1])
    # 画图时，为了方便对照数据集的标签序号，类别从1开始
    targets = targets + 1
    targets = np.where(targets == 2, 31, targets)
    targets = np.where(targets == 3, 32, targets)
    vertices = process_vertice_by_ori_bound(vertices=vertices,ori_bound=ori_bound)
    trimesh = {
        'vertices': vertices,
        'faces': faces,
    }
    plot_trimesh_with_labels(trimesh, targets, save_path=f"test_gt_mesh.png")


    # 提取多边形
    merged_solution = MergePolygonSolution()
    merged_solution.load_data_from_model_inference(
        vertices=vertices,
        faces=faces,
        labels=targets,
    )
    merged_solution.start_work()
    merged_rooms = merged_solution.get_merged_polygons_with_coords()
    plot_floorplan_with_rooms_and_bound(merged_rooms,ori_bound,save_path=f"test_merged_rooms.png")
    # predict_rooms = merged_solution.get_merged_polygons_as_raster_format()
    #
    # density_folder = config.gt_coco_path
    # density_folder = r'G:\workspace_plane2DDL\augment_point_cloud_density'
    # mock_img_folder = os.path.join(density_folder, 'test')
    # mock_json_path = os.path.join(density_folder, 'annotations', 'test.json')
    #
    # density, gt_polygons_list = parse_coco_dict(
    #     json_path=mock_json_path,
    #     img_path=mock_img_folder,
    #     num_scenes=scene_num,
    # )
    # gt_data = {
    #     'density':density,
    #     'polygons_list':gt_polygons_list
    # }
    #
    # cur_scene_eval = Evaluator(gt_data)
    # quan_result_dict =cur_scene_eval.evaluate_scene(predict_rooms)
    # print(quan_result_dict)
    #
    # plot_pred = True
    # plot_density = True
    # room_polys = predict_rooms
    #
    # if plot_pred:
    #     # plot regular room floorplan # 绘制纯矢量图（不带密度图背景）
    #     room_polys = [np.array(r) for r in room_polys]
    #     floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
    #     cv2.imwrite('{}_pred_floorplan.png'.format(scene_name), floorplan_map)
    # if plot_density:
    #     # density shape: (256, 256)
    #     density_map = np.expand_dims(density, axis=-1)
    #     density_map = np.repeat(density_map, 3, axis=2)
    #     pred_room_map = np.zeros([256, 256, 3])
    #
    #     for room_poly in room_polys:
    #         pred_room_map = plot_room_map(room_poly, pred_room_map)
    #
    #     # plot predicted polygon overlaid on the density map
    #     pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
    #     cv2.imwrite('{}_pred_room_map.png'.format(scene_name), pred_room_map)

    # plot_ground_truth_and_prediction(
    #     vertices=vertices, faces=faces,
    #     gt_labels=targets, pred_labels=targets,
    #     output_path=r"test.png")
    # vertices = process_vertice_by_ori_bound(
    #     ori_bound=ori_bound, vertices=vertices)
    # export_mesh_to_shp(
    #     vertices=vertices, faces=faces,labels=targets,
    #     output_path=f"test_shpfile_predict")
    print("done")


def test_tmp():
    time_record = []
    for i in range(5):
        start = time.perf_counter()
        a = 5
        end = time.perf_counter()
        time_record.append(end-start)
    # time = [2.0,2.1,2.2,2.3]
    avg_time = sum(time_record)/len(time_record)
    d = {}
    d['avg_time'] = avg_time
    json_str = json.dumps(d)
    print(json_str)

if __name__ == '__main__':
    test_tmp()