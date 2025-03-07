import os
import sys

import cv2
import numpy as np
from shapely import Polygon

from srcipt.generate_coco_stru3d import parse_coco_dict
from util.Evaluator import Evaluator
from util.merge_polygon import MergePolygonSolution
from util.visualization import plot_floorplan_with_regions, plot_room_map

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import hydra
from dataset.floorplan_triangles import FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes
from trainer.train_triangle import GraphTransformerEncoder
from tqdm import tqdm


def eval_model(config, load_checkpoint_path, dataset,plot_pred=False, plot_density=False):
    save_inference_path = config.save_inference_path
    # checkpoint_path = os.path.join(load_model_experiment_path, "checkpoints", load_checkpoint_name)

    # if save_path is None, use the same dir as the checkpoint
    if save_inference_path is None:
        load_checkpoint_root = os.path.dirname(load_checkpoint_path)
        checkpoint_name = os.path.basename(load_checkpoint_path).split(".")[0]
        save_inference_path = os.path.join(load_checkpoint_root, f"inference_from_{checkpoint_name}")
    predict_path_root = save_inference_path  # rename

    print(predict_path_root)
    if not os.path.exists(predict_path_root):
        os.makedirs(predict_path_root)
        print(f"make dir {predict_path_root}")

    # model = TriangleTokenizationGraphConv(config)
    model = GraphTransformerEncoder.load_from_checkpoint(checkpoint_path=load_checkpoint_path)
    model.eval()
    total_task_num = len(dataset)
    quant_result_dict = None
    scene_counter = 0
    progress_bar = tqdm(total=total_task_num, desc="Inference")
    error_scene = []
    for idx in range(total_task_num):
        data = dataset.get(idx)
        # 训练时，为了方便索引，类别从0开始
        _, targets, vertices, faces, _, op = dataset.get_all_features_for_shape(idx)
        scene_name = dataset.get_name(idx)
        scene_num = int(scene_name.split("_")[1])
        predict = model.inference_data(data)
        predict = np.where(predict == 2, 31, predict)
        predict = np.where(predict == 3, 32, predict)

        # 提取多边形
        merged_solution = MergePolygonSolution()
        try:
            merged_solution.load_data_from_model_inference(vertices, faces, predict)
            merged_solution.start_work()
            predict_rooms = merged_solution.get_merged_polygons_as_raster_format()
        except Exception as e:
            print(f"scene_name:{scene_name} failed to reconstruct polygons.error:{e}")
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

        progress_bar.update(1)


@hydra.main(config_path='../config', config_name='graph_transformer', version_base='1.2')
def main(config):
    load_checkpoint_path = config.load_checkpoint_path
    if config.inference_dataset_path is None:
        print("inference_dataset_path:None，user test part of train model dataset")
        dataset = FPTriangleWithGeneratedFeaturesAndLabel3ClsNodes(config, 'test')
    else:
        print(f"inference_dataset_path:{config.inference_dataset_path},but currently not support across dataset inference")
        return
    # dataset = build_dataset_for_cross("/mnt/data2/pengyan/augment_stru3d_featured_shp_0-250",config)
    eval_model(config, load_checkpoint_path, dataset, plot_pred=True, plot_density=True)
