import json
import os
import pickle
import shutil
from pathlib import Path

import cv2
import trimesh
from shapely import Polygon
from tqdm import tqdm
from trimesh.exchange import obj
import numpy as np
import time
from ablation.only_segment_room_and_wall import FPTriangleWithThreeClsNodes
from dataset import sort_vertices_and_faces
from dataset.floorplan_triangles import FPTriangleNodes,FPOriginTriangleNodes
from srcipt.generate_coco_stru3d import parse_coco_dict
from util.Evaluator import Evaluator
from util.eval_rms import RMS_Evaluator
from util.merge_polygon import MergePolygonSolution
from util.s3d_data_load import read_s3d_mesh_info, get_faces_data, get_no_confidence_faces, get_vertices_data, \
    get_vertices_coord
from util.s3d_data_process import process_vertice_by_op_record, process_vertice_by_ori_bound, OBJExporter
from util.visualization import plot_trimesh_with_labels, plot_floorplan_with_rooms_and_bound, export_mesh_to_shp, \
    plot_floorplan_with_regions
import hydra
from dataset.floorplan_triangles import FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes
from util.misc import scale_vertices, normalize_vertices, shift_vertices


def inject_core(config,mock_file):
    scene_name = os.path.basename(mock_file).split('.')[0]
    # face,predict = get_no_confidence_faces(predict_path)
    config.scale_augment = False
    config.shift_augment = False
    config.dataset_root = r"G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\shp_root" # 测试数据集位置
    dataset = FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes(config,'test',split_mode="scene_names")
    n = len(dataset)
    idx = 0
    while idx < n:
        check_scene_name = dataset.get_name(idx)
        if check_scene_name == scene_name:
            break
        idx += 1
    if idx == n:
        print(f"scene {scene_name} not found in dataset")
        return
    _, targets, vertices, faces, _, ori_bound = dataset.get_all_features_for_shape(idx)
    return vertices,faces

def inject(mock_data,config,mock_file):
    vertices,faces = inject_core(config,mock_file)
    export_mesh_to_shp(vertices=vertices,faces=faces,labels=mock_data,output_path=mock_file)


def inject_and_recover_to_real_world_scale(mock_data,config,mock_file,real_world_ref_json):
    vertices,faces = inject_core(config,mock_file)

    # 读取真实世界的参考json文件
    with open(real_world_ref_json, 'r') as f:
        real_world_ref = json.load(f)
    current_scene_name = os.path.basename(mock_file).split('.')[0]

    # 找到真实世界的参考点
    real_world_info = real_world_ref[current_scene_name]
    real_world_x_range = real_world_info[0]
    real_world_bottom_left_point = [real_world_info[2], real_world_info[3],0.0]

    # 计算真实世界的尺度
    real_world_scale = real_world_x_range / (vertices[:, 0].max() - vertices[:, 0].min())
    real_world_offset = vertices.min(axis=0)

    # 还原到真实世界的尺度
    vertices = vertices - real_world_offset
    vertices = vertices * real_world_scale
    vertices = vertices + np.array(real_world_bottom_left_point)

    export_mesh_to_shp(vertices=vertices,faces=faces,labels=mock_data,output_path=mock_file)

def plot(mock_file):
    room_polys, scene_name = merge_polygon(mock_file,is_standard_room=False)
    room_polys = [np.array(r) for r in room_polys]
    floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
    cv2.imwrite(os.path.join(mock_file, '{}_pred_floorplan.png'.format(scene_name)), floorplan_map)


def plot_original_aspect_raito_labeled_triangle_mesh(mock_file):
    # 从shp文件从提取顶点数据、面数据、标签数据
    scene_name = os.path.basename(mock_file).split('.')[0]
    vertices_file = os.path.join(mock_file, f"vertexes.shp")
    faces_file = os.path.join(mock_file, f"poly.shp")
    vertices = get_vertices_coord(vertices_file)
    faces, predict = get_no_confidence_faces(faces_file)

    triangle_mesh = {
        'vertices': vertices,
        'faces': faces,
    }

    plot_trimesh_with_labels(trimesh=triangle_mesh, labels=predict,
                             save_path=os.path.join(mock_file, f"{scene_name}_trimesh.png"))


def merge_polygon(mock_file,is_standard_room=True):
    scene_name = os.path.basename(mock_file).split('.')[0]
    vertices_file = os.path.join(mock_file, f"vertexes.shp")
    faces_file = os.path.join(mock_file, f"poly.shp")
    vertices = get_vertices_coord(vertices_file)
    faces, predict = get_no_confidence_faces(faces_file)
    # 提取多边形
    merged_solution = MergePolygonSolution()
    merged_solution.load_data_from_model_inference(
        vertices=vertices,
        faces=faces,
        labels=predict,
    )
    merged_solution.start_work()
    if is_standard_room:
        predict_rooms = merged_solution.get_merged_polygons_with_coords()
        return predict_rooms, scene_name
    else:
        predict_rooms = merged_solution.get_merged_polygons_as_raster_format()
        room_polys = []
        # 筛选预测房间
        for polygon in predict_rooms:
            corners = polygon  # rename polygon as corners(因为顶点集构成了多边形)
            # only regular rooms
            if len(corners) >= 4 and Polygon(corners).area >= 100:
                room_polys.append(corners)
        return room_polys, scene_name



def merge_polygon_and_output_as_obj(mock_file,obj_file=None):
    if obj_file is None:
        obj_file = os.path.join(mock_file, f"predict_rooms.obj")
    scene_name = os.path.basename(mock_file).split('.')[0]
    vertices_file = os.path.join(mock_file, f"vertexes.shp")
    faces_file = os.path.join(mock_file, f"poly.shp")
    vertices = get_vertices_coord(vertices_file)
    faces, predict = get_no_confidence_faces(faces_file)
    # 提取多边形
    merged_solution = MergePolygonSolution()
    merged_solution.load_data_from_model_inference(
        vertices=vertices,
        faces=faces,
        labels=predict,
    )
    merged_solution.start_work()
    predict_rooms = merged_solution.get_merged_polygons_with_coords()

    print(f"predict_rooms:{predict_rooms}")

    exporter = OBJExporter()
    for i in range(len(predict_rooms)):
        polygon = predict_rooms[i]
        polygon_obj = []
        for i in range(len(polygon)):
            point = polygon[i]
            polygon_obj.append([point[0], point[1], 0])
        exporter.add_polygon(polygon_obj)

    exporter.export(obj_file)
    return predict_rooms


def eval_data(mock_file):
    # 读取coco格式的gt数据
    gt_root = r"G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\coco_root"
    gt_img_folder = os.path.join(gt_root, 'test')
    gt_json_path = os.path.join(gt_root, 'annotations', 'test.json')
    scene_name = os.path.basename(mock_file).split('.')[0]
    scene_num = int(scene_name.split('_')[-1])

    # 依据场景名称寻找coco格式的gt数据
    density, gt_polygons_list = parse_coco_dict(
        json_path=gt_json_path,
        img_path=gt_img_folder,
        num_scenes=scene_num
    )
    gt_data = {
        'density': density,
        'polygons_list': gt_polygons_list
    }
    room_polys, _ = merge_polygon(mock_file)
    # 生成eval实体
    cur_scene_eval = Evaluator(gt_data)
    # 评价当前预测
    quant_result_dict_scene = cur_scene_eval.evaluate_scene(room_polys)
    print(f"scene {scene_name} evaluation result: {quant_result_dict_scene}")


def extra_eval_rms(mock_file):
    predict_rooms = merge_polygon_and_output_as_obj(mock_file)
    las_root = r"G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\pc_root"
    scene_name = os.path.basename(mock_file).split('.')[0]
    las_path = os.path.join(las_root, scene_name, f"poinc_cloud_part.las")
    evaluator = RMS_Evaluator(
        las_path=las_path,
        floor_plan_data=predict_rooms,
        output_las_result_path=os.path.join(mock_file, f"{scene_name}_rms_result.las"),
    )
    evaluator.calculate_rms()


@hydra.main(config_path='../config', config_name='graph_transformer', version_base='1.2')
def main_2(config):
    predict_path = r"G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\inference_from_98-0/predict.json"
    mock_root = r"G:\workspace_plane2DDL\testData\mock_data"
    scene_num = 2
    scene_name = f"scene_{scene_num:03d}"

    mock_file = os.path.join(mock_root, scene_name)
    with open(predict_path, 'r') as f:
        predict = json.load(f)
    mock_data = predict[scene_name]
    # inject(mock_data,config,mock_file)
    # real_world_ref_json = r"G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\dataset_real_world_range.json"
    # inject_and_recover_to_real_world_scale(mock_data,config,mock_file,real_world_ref_json)
    # plot_original_aspect_raito_labeled_triangle_mesh(mock_file)
    # eval_data(mock_file)
    # extra_eval_rms(mock_file)
    plot(mock_file)
    # _, targets, vertices, faces, _, ori_bound = dataset.get_all_features_for_shape(idx)
    # scene_name = dataset.get_name(idx)
    # print(f"scene_name:{scene_name}")
    # output_path = os.path.join(f"{scene_name}_shpfile_predict")
    # if os.path.exists(output_path):
    #     shutil.rmtree(output_path)
    # os.makedirs(output_path)
    #
    # vertices = process_vertice_by_ori_bound(vertices=vertices, ori_bound=ori_bound)
    # trimesh = {
    #     'vertices': vertices,
    #     'faces': faces,
    # }
    # plot_trimesh_with_labels(trimesh, predict, save_path=os.path.join(output_path, "test_gt_mesh.png"))
    # # 提取多边形
    # merged_solution = MergePolygonSolution()
    # merged_solution.load_data_from_model_inference(
    #     vertices=vertices,
    #     faces=faces,
    #     labels=predict,
    # )
    # merged_solution.start_work()
    # standard_polygons = merged_solution.get_merged_polygons_with_coords()
    # predict_rooms = merged_solution.get_merged_polygons_as_raster_format()
    # # plot floorplan
    # plot_floorplan_with_rooms_and_bound(standard_polygons, ori_bound,
    #                                     save_path=os.path.join(output_path,
    #                                                            f"{scene_name}_floorplan.png"))
    #
    # plot_pred = True
    # # plot_density = True
    # room_polys = predict_rooms
    # #
    # if plot_pred:
    #     # plot regular room floorplan # 绘制纯矢量图（不带密度图背景）
    #     room_polys = [np.array(r) for r in room_polys]
    #     floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
    #     cv2.imwrite(os.path.join(output_path,'{}_pred_floorplan.png'.format(scene_name)), floorplan_map)
    #
    #
    # # plot_floorplan_with_rooms_and_bound(merged_rooms,ori_bound,save_path=os.path.join(output_path, "test_merged_rooms.png"))
    #


@hydra.main(config_path='../config', config_name='graph_transformer', version_base='1.2')
def main(config):
    # r'G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_bbox_10_percent_shp'
    config.dataset_root = r"G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\shp_root" # 测试数据集位置
    config.scale_augment = False
    config.shift_augment = False
    predict_dict_path = r'G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\inference_from_98-0/predict.json'
    predict_dict = json.load(open(predict_dict_path, 'r'))
    dataset = FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes(config,'test',split_mode="scene_names")

    n = len(dataset)
    print(f"dataset len:{n}")
    print(f"start:{dataset.get_name(0)}")
    print(f"end:{dataset.get_name(n-1)}")
    idx = 3
    _, targets, vertices, faces, _, ori_bound = dataset.get_all_features_for_shape(idx)
    scene_name = dataset.get_name(idx)
    print(f"scene_name:{scene_name}")
    scene_num = int(scene_name.split('_')[-1])
    predict = predict_dict[scene_name]
    output_path = os.path.join(f"{scene_name}_shpfile_predict")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    vertices = process_vertice_by_ori_bound(vertices=vertices, ori_bound=ori_bound)
    trimesh = {
        'vertices': vertices,
        'faces': faces,
    }
    plot_trimesh_with_labels(trimesh, predict, save_path=os.path.join(output_path, "test_gt_mesh.png"))
    # export_mesh_to_shp(vertices=vertices, faces=faces,\
    #                    labels=predict, output_path=os.path.join(f"{scene_name}_shpfile_predict"))

    # 提取多边形
    merged_solution = MergePolygonSolution()
    merged_solution.load_data_from_model_inference(
        vertices=vertices,
        faces=faces,
        labels=predict,
    )
    merged_solution.start_work()
    predict_rooms = merged_solution.get_merged_polygons_as_raster_format()

    # plot_floorplan_with_rooms_and_bound(merged_rooms,ori_bound,save_path=os.path.join(output_path, "test_merged_rooms.png"))

    # plot regular room floorplan # 绘制纯矢量图（不带密度图背景）
    room_polys = [np.array(r) for r in predict_rooms]
    floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
    cv2.imwrite(os.path.join(output_path, '{}_pred_floorplan.png'.format(scene_name)), floorplan_map)

    gt_root = r'G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\coco_root'
    gt_img_folder = os.path.join(gt_root, 'test')
    gt_json_path = os.path.join(gt_root, 'annotations', 'test.json')

    # 依据场景名称寻找coco格式的gt数据
    density, gt_polygons_list = parse_coco_dict(
        json_path=gt_json_path,
        img_path=gt_img_folder,
        num_scenes=scene_num
    )

    gt_data = {
        'density': density,
        'polygons_list': gt_polygons_list
    }

    # 生成eval实体
    cur_scene_eval = Evaluator(gt_data)
    quant_result_dict_scene = cur_scene_eval.evaluate_scene(room_polys)
    print(quant_result_dict_scene)


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
    plot_pred = True
    # plot_density = True
    # room_polys = predict_rooms
    #
    if plot_pred:
        # plot regular room floorplan # 绘制纯矢量图（不带密度图背景）
        room_polys = [np.array(r) for r in room_polys]
        floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
        cv2.imwrite('{}_pred_floorplan.png'.format(scene_name), floorplan_map)
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



@hydra.main(config_path='../config', config_name='graph_transformer', version_base='1.2')
def predict_from_label_to_room(config):
    # label
    label_predict_json = r"G:\workspace_plane2DDL\record\s3d_ours_reference\predict.json"
    with open(label_predict_json, 'r') as f:
        predict_label_info = json.load(f)
    # scale_file
    scale_file = r"G:\workspace_plane2DDL\real_point_cloud_dataset\full_test_scale_record\dataset_real_world_range.json"
    with open(scale_file, 'r') as f:
        scale_info = json.load(f)

    # save_root
    save_root = r"G:\workspace_plane2DDL\record\s3d_ours_reference"


    scene_names = predict_label_info.keys()
    # dataset
    config.scale_augment = False
    config.shift_augment = False
    config.dataset_root = r"G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_bbox_10_percent_shp" # 测试数据集位置
    dataset = FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes(config,'test',split_mode="scene_names")

    predict_scenes = {}
    bar = tqdm(total = len(scene_names))
    for idx in range(len(dataset)):
        scene_name = dataset.get_name(idx)
        print(f"scene_name:{scene_name}")
        if scene_name not in scene_names:
            continue
        _, targets, vertices, faces, _, ori_bound = dataset.get_all_features_for_shape(idx)

        vertices = process_vertice_by_ori_bound(ori_bound=ori_bound, vertices=vertices)
        cur_label = predict_label_info[scene_name]
        # 提取多边形
        merged_solution = MergePolygonSolution()
        merged_solution.load_data_from_model_inference(
            vertices=vertices,
            faces=faces,
            labels=cur_label,
        )
        merged_solution.start_work()
        predict_rooms = merged_solution.get_merged_polygons_with_coords()

        predict_scenes[scene_name] = predict_rooms
        # 绘制结果
        obj_file = os.path.join(save_root, f"{scene_name}_pred_rooms.obj")
        exporter = OBJExporter()
        for i in range(len(predict_rooms)):
            polygon = predict_rooms[i]
            polygon_obj = []
            for i in range(len(polygon)):
                point = polygon[i]
                polygon_obj.append([point[0], point[1], 0])
            exporter.add_polygon(polygon_obj)

        exporter.export(obj_file)
        bar.update(1)
    bar.close()

    with open(os.path.join(save_root, f"predict_scenes.json"), 'w') as f:
        json.dump(predict_scenes, f)



if __name__ == '__main__':
    main_2()
    # predict_from_label_to_room()