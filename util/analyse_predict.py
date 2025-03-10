import json
import os

import cv2
import numpy as np
from shapely import Polygon
from tqdm import tqdm

from dataset.floorplan_triangles import FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes
from srcipt.generate_coco_stru3d import parse_coco_dict
from util.Evaluator import Evaluator
from util.merge_polygon import MergePolygonSolution
from util.s3d_data_process import process_vertice_by_ori_bound
import hydra

from util.visualization import plot_floorplan_with_regions, plot_room_map, plot_trimesh_with_labels, \
    plot_floorplan_with_rooms_and_bound


def eval_core(config, predict_json_path, dataset,plot_pred=False, plot_density=False,plot_gt=False):
    predict_path_root = predict_json_path # 同名
    predict_json = os.path.join(predict_path_root, 'predict.txt')
    predict_dict = json.load(open(predict_json, 'r'))

    trimesh_predict_path = os.path.join(predict_path_root, "trimesh_predict")
    if not os.path.exists(trimesh_predict_path):
        os.makedirs(trimesh_predict_path)

    error_polygon_construct_path = os.path.join(predict_path_root, "error_polygon_construct")
    if not os.path.exists(error_polygon_construct_path):
        os.makedirs(error_polygon_construct_path)

    total_task_num = len(predict_dict.keys())
    quant_result_dict = None
    scene_counter = 0
    progress_bar = tqdm(total=total_task_num, desc="Inference")
    error_scene = []
    time_record = []
    for idx in range(total_task_num):

        data = dataset.get(idx)
        # 训练时，为了方便索引，类别从0开始
        _, targets, vertices, faces, _, ori_bound = dataset.get_all_features_for_shape(idx)
        scene_name = dataset.get_name(idx)
        scene_num = int(scene_name.split("_")[1])
        progress_bar.set_description(f"processing scene: {scene_name} #idx{idx}")
        predict = predict_dict[scene_name]
        predict = np.array(predict)
        predict = np.where(predict == 4, 32, predict)
        vertices = process_vertice_by_ori_bound(vertices=vertices, ori_bound=ori_bound)
        trimesh = {
            'vertices': vertices,
            'faces': faces,
        }
        # plot_trimesh_with_labels(trimesh, targets, save_path=f"test_gt_mesh.png")

        # 提取多边形
        merged_solution = MergePolygonSolution()
        try:
            merged_solution.load_data_from_model_inference(vertices, faces, predict)
            merged_solution.start_work()
            standard_polygons = merged_solution.get_merged_polygons_with_coords()
            predict_rooms = merged_solution.get_merged_polygons_as_raster_format()
        except Exception as e:
            print(f"scene_name:{scene_name} failed to reconstruct polygons.error:{e}")

            # plot trimesh
            plot_trimesh_with_labels(trimesh=trimesh, labels=predict,
                                     save_path=os.path.join(error_polygon_construct_path, f"{scene_name}_trimesh.png"))

            error_scene.append(scene_name)
            progress_bar.update(1)
            continue

        # 读取coco格式的gt数据
        gt_root = config.gt_coco_path
        gt_img_folder = os.path.join(gt_root, 'test')
        gt_json_path = os.path.join(gt_root, 'annotations', 'test.json')

        # 依据场景名称寻找coco格式的gt数据
        density,gt_polygons_list = parse_coco_dict(
            json_path = gt_json_path,
            img_path = gt_img_folder,
            num_scenes = scene_num
        )
        gt_data ={
            'density':density,
            'polygons_list':gt_polygons_list
        }

        # 生成eval实体
        cur_scene_eval = Evaluator(gt_data)

        room_polys = []
        # 筛选预测房间
        for polygon in predict_rooms:
            corners = polygon # rename polygon as corners(因为顶点集构成了多边形)
            # only regular rooms
            if len(corners) >= 4 and Polygon(corners).area >= 100:
                room_polys.append(corners)

        # 评价当前预测
        quant_result_dict_scene = cur_scene_eval.evaluate_scene(room_polys)

        if quant_result_dict is None:
            quant_result_dict = quant_result_dict_scene
        else:
            for k in quant_result_dict.keys():
                quant_result_dict[k] += quant_result_dict_scene[k]

        scene_counter += 1

        if plot_pred:
            # plot regular room floorplan # 绘制纯矢量图（不带密度图背景）
            room_polys = [np.array(r) for r in room_polys]
            floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
            cv2.imwrite(os.path.join(predict_path_root, '{}_pred_floorplan.png'.format(scene_name)), floorplan_map)
        if plot_density:
            # density shape: (256, 256)
            density_map = np.expand_dims(density, axis= -1)
            density_map = np.repeat(density_map, 3, axis=2)
            pred_room_map = np.zeros([256, 256, 3])

            for room_poly in room_polys:
                pred_room_map = plot_room_map(room_poly, pred_room_map)

            # plot predicted polygon overlaid on the density map
            pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
            cv2.imwrite(os.path.join(predict_path_root, '{}_pred_room_map.png'.format(scene_name)), pred_room_map)
        if plot_gt:
            # plot regular room floorplan # 绘制纯矢量图（不带密度图背景）
            room_polys = [np.array(r) for r in gt_polygons_list]
            floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
            cv2.imwrite(os.path.join(predict_path_root, '{}_gt_floorplan.png'.format(scene_name)), floorplan_map)
        if idx % 10 == 0:
            # plot trimesh
            plot_trimesh_with_labels(trimesh=trimesh, labels=predict,
                                     save_path=os.path.join(trimesh_predict_path, f"{scene_name}_trimesh.png"))
            # plot floorplan
            plot_floorplan_with_rooms_and_bound(standard_polygons, ori_bound,
                                                save_path=os.path.join(trimesh_predict_path, f"{scene_name}_floorplan.png"))
        progress_bar.update(1)

    # 求平均
    for k in quant_result_dict.keys():
        quant_result_dict[k] /= float(scene_counter)

    metric_category = ['room','corner','angles']

    for metric in metric_category:
        prec = quant_result_dict[metric+'_prec']
        rec = quant_result_dict[metric+'_rec']
        f1 = 2*prec*rec/(prec+rec)
        quant_result_dict[metric+'_f1'] = f1

    quant_result_dict['time_avg'] = sum(time_record)/len(time_record)

    print("*************************************************")
    print(quant_result_dict)
    print(f"error_scene:{error_scene}")
    print("*************************************************")

    with open(os.path.join(predict_path_root, 'quant_results.txt'), 'w') as file:
        file.write(json.dumps(quant_result_dict))


def debug_eval_core(config, predict_json_path, dataset,plot_pred=False, plot_density=False):
    idx = 103
    predict_path_root = predict_json_path  # 同名
    predict_json = os.path.join(predict_path_root, 'predict.txt')
    predict_dict = json.load(open(predict_json, 'r'))

    trimesh_predict_path = os.path.join(predict_path_root, "trimesh_predict")
    if not os.path.exists(trimesh_predict_path):
        os.makedirs(trimesh_predict_path)

    error_polygon_construct_path = os.path.join(predict_path_root, "error_polygon_construct")
    if not os.path.exists(error_polygon_construct_path):
        os.makedirs(error_polygon_construct_path)

    total_task_num = len(predict_dict.keys())
    quant_result_dict = None
    scene_counter = 0
    progress_bar = tqdm(total=total_task_num, desc="Inference")
    error_scene = []
    time_record = []


    # 训练时，为了方便索引，类别从0开始
    _, targets, vertices, faces, _, ori_bound = dataset.get_all_features_for_shape(idx)
    scene_name = dataset.get_name(idx)
    scene_num = int(scene_name.split("_")[1])
    progress_bar.set_description(f"Inference {scene_name}")
    predict = predict_dict[scene_name]
    predict = np.array(predict)
    predict = np.where(predict == 4, 32, predict)
    vertices = process_vertice_by_ori_bound(vertices=vertices, ori_bound=ori_bound)
    trimesh = {
        'vertices': vertices,
        'faces': faces,
    }
    # plot_trimesh_with_labels(trimesh, targets, save_path=f"test_gt_mesh.png")

    # 提取多边形
    merged_solution = MergePolygonSolution()
    try:
        merged_solution.load_data_from_model_inference(vertices, faces, predict)
        merged_solution.start_work_with_time_analysis()
        standard_polygons = merged_solution.get_merged_polygons_with_coords()
        predict_rooms = merged_solution.get_merged_polygons_as_raster_format()
    except Exception as e:
        print(f"scene_name:{scene_name} failed to reconstruct polygons.error:{e}")

        # plot trimesh
        plot_trimesh_with_labels(trimesh=trimesh, labels=predict,
                                 save_path=os.path.join(error_polygon_construct_path, f"{scene_name}_trimesh.png"))

        error_scene.append(scene_name)
        progress_bar.update(1)



@hydra.main(config_path='../config', config_name='graph_transformer', version_base='1.2')
def main(config):
    dataset = FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes(config, 'test','scene_names')
    result_path = r"G:\workspace_plane2DDL\to_construct\augment\0308"
    # debug_eval_core(config, result_path, dataset,plot_pred=True, plot_density=True)
    eval_core(config, result_path, dataset,plot_pred=True, plot_density=True,plot_gt=True)


if __name__ == '__main__':
    main()