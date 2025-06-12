import json
import os

import numpy as np
from shapely import Polygon

from srcipt.generate_coco_stru3d import parse_coco_dict
from util.Evaluator import Evaluator
from util.eval_rms import RMS_Evaluator
from util.s3d_data_process import from_density_map_result_to_real_world_and_save_in_obj

PREDICT_POLYGON_INFO_JSON = r'...\real_point_cloud_dataset\other_point_cloud\jys_result\jys_predict_rooms_density.json'
# PREDICT_POLYGON_INFO_JSON = r'G:\workspace_plane2DDL\testData\mock_data\scd_predict_rooms.json'
OUTPUT_FOLDER = r'...\real_point_cloud_dataset\other_point_cloud\jys_result'
# 真值（COCO格式）
COCO_PATH = r"...\real_point_cloud_dataset\other_point_cloud\coco_root"


def predict_info_adjust(predict_polygon_infos):
    result = {}
    for info in predict_polygon_infos:
        image_id = info['image_id']
        scene_name = f"scene_{image_id:03d}"
        if result.get(scene_name) is None:
            result[scene_name] = []
        points = info['segmentation'][0]
        points = [points[i:i + 2] for i in range(0, len(points), 2)]
        result[scene_name].append(np.array(points).astype(np.int32))

    return result

def predict_info_adjust_v2(predict_polygon_infos):
    result = {}
    for scene_name in predict_polygon_infos.keys():
        rooms = predict_polygon_infos[scene_name]
        result[scene_name] = [np.array(r) for r in rooms]
    return result


def main(specified_scene_name=None):
    predict_polygon_info_json = PREDICT_POLYGON_INFO_JSON
    with open(predict_polygon_info_json, 'r') as f:
        predict_polygon_infos = json.load(f)
    # predict_polygon_infos = predict_info_adjust(predict_polygon_infos)
    predict_polygon_infos = predict_info_adjust_v2(predict_polygon_infos)

    coco_format_gt_path = COCO_PATH
    gt_img_folder = os.path.join(coco_format_gt_path, 'test')
    gt_json_path = os.path.join(coco_format_gt_path, 'annotations', 'test.json')

    scene_names = predict_polygon_infos.keys()
    for scene_name in scene_names:
        if type(specified_scene_name) == list:
            if scene_name not in specified_scene_name:
                continue
        elif type(specified_scene_name) == str:
            if specified_scene_name != scene_name:
                continue

        scene_num = int(scene_name.split("_")[1])
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
        cur_scene_evaler = Evaluator(gt_data)
        predict_rooms = predict_polygon_infos[scene_name]

        room_polys = []
        # 筛选预测房间
        for polygon in predict_rooms:
            corners = polygon  # rename polygon as corners(因为顶点集构成了多边形)
            # only regular rooms
            if len(corners) >= 4 and Polygon(corners).area >= 100:
                room_polys.append(corners)

        quant_result = cur_scene_evaler.evaluate_scene(room_polys)
        print(scene_name + 'RMS: ', quant_result)
        output_path = os.path.join(OUTPUT_FOLDER, f"f1_score_{scene_name}.json")
        with open(output_path, 'w') as f:
            json.dump(quant_result, f)


def collect_and_calculate_f1_score():
    files = os.listdir(OUTPUT_FOLDER)
    avg_result = {}
    element_num = 0
    for file in files:
        if not file.startswith("f1_score_scene_"):
            continue

        with open(os.path.join(OUTPUT_FOLDER, file), 'r') as f:
            quant_result = json.load(f)
        for key in quant_result.keys():
            if avg_result.get(key) is None:
                avg_result[key] = quant_result[key]
            else:
                avg_result[key] += quant_result[key]
        element_num += 1

    for key in avg_result.keys():
        avg_result[key] /= element_num

    f1_list = ['room','corner','angles']
    for f1_type in f1_list:
        precision = avg_result[f1_type+'_prec']
        recall = avg_result[f1_type+'_rec']
        f1_score = 2*precision*recall/(precision+recall)
        avg_result[f1_type+'_f1'] = f1_score

    avg_result_path = os.path.join(OUTPUT_FOLDER, "avg_f1_score.json")
    with open(avg_result_path, 'w') as f:
        json.dump(avg_result, f)


    print("avg_result: ", avg_result)



if __name__ == '__main__':
    main()
    collect_and_calculate_f1_score()