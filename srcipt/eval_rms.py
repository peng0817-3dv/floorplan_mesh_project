import json
import os

from util.eval_rms import RMS_Evaluator
from util.s3d_data_process import from_density_map_result_to_real_world_and_save_in_obj, \
    from_density_map_result_to_real_world
from tqdm import tqdm

# PREDICT_POLYGON_INFO_JSON = r'G:\workspace_plane2DDL\record\s3d++推理结果\fri_npy\fri_result.json'
# PREDICT_POLYGON_INFO_JSON = r'G:\workspace_plane2DDL\testData\mock_data\scd_predict_rooms.json'
# PREDICT_POLYGON_INFO_JSON = r'G:\workspace_plane2DDL\record\scd数据集推理结果\jys\jys_predict_rooms_2.json'
PREDICT_POLYGON_INFO_JSON = r'G:\workspace_plane2DDL\record\scd数据集推理结果\fri\results_off\layer0.json'


POINT_CLOUD_ROOT = r'G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\pc_root'
OUTPUT_FOLDER = r'G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\frinet_result'
# S3D_REAL_WORLD_INFO_JSON = r'G:\workspace_plane2DDL\real_point_cloud_dataset\augment_stru3d_real_world_info.json'
S3D_REAL_WORLD_INFO_JSON = r'G:\workspace_plane2DDL\real_point_cloud_dataset\other_point_cloud\dataset_real_world_range.json'

def raster_to_real_world():
    pass


def predict_info_adjust(before):
    adjust_infos = {}
    for info in before:
        image_id = info['image_id']
        scene_name = f"scene_{image_id:03d}"
        if adjust_infos.get(scene_name) is None:
            adjust_infos[scene_name] = []
        points = info['segmentation'][0]
        room = [[points[i], points[(i+1)%len(points)]] for i in range(0, len(points), 2)]
        adjust_infos[scene_name].append(room)
    return adjust_infos


def main(is_from_raster = False,is_from_layer0=False,specified_scene_name=None):
    predict_polygon_info_json = PREDICT_POLYGON_INFO_JSON
    with open(predict_polygon_info_json, 'r') as f:
        predict_polygon_infos = json.load(f)

    s3d_real_world_info_json = S3D_REAL_WORLD_INFO_JSON
    with open(s3d_real_world_info_json, 'r') as f:
        s3d_real_world_infos = json.load(f)

    if is_from_raster and is_from_layer0:
        adjust_infos = predict_info_adjust(predict_polygon_infos)
        predict_polygon_infos = adjust_infos

    point_cloud_root = POINT_CLOUD_ROOT
    scene_names = predict_polygon_infos.keys()
    rms_result_visualization_folder = OUTPUT_FOLDER
    if not os.path.exists(rms_result_visualization_folder):
        os.makedirs(rms_result_visualization_folder)
    rms_result_record_file = os.path.join(rms_result_visualization_folder, 'rms_result.json')
    rms_result_record = {}
    rms_values = []
    scale_scenes = s3d_real_world_infos.keys()

    bar = tqdm(total =len(scene_names))
    # save_las_sample = ["scene_03250","scene_03251","scene_03252"]
    for scene_name in scene_names:
        bar.update(1)
        if type(specified_scene_name) == list:
            if scene_name not in specified_scene_name:
                continue
        elif type(specified_scene_name) == str:
            if specified_scene_name != scene_name:
                continue

        if not scene_name.startswith('scene_'):
            continue
        if scene_name not in scale_scenes:
            continue

        save_las_flag = True
        # if scene_name in save_las_sample:
        #     save_las_flag = True


        predict_polygon_info = predict_polygon_infos[scene_name]

        if is_from_raster:
            s3d_bound_info = s3d_real_world_infos[scene_name]
            if save_las_flag:
                predict_polygon_info = from_density_map_result_to_real_world_and_save_in_obj(
                    density_map_rooms= predict_polygon_info,
                    x_range= s3d_bound_info[0],
                    y_range= s3d_bound_info[1],
                    x_min= s3d_bound_info[2],
                    y_min= s3d_bound_info[3],
                    obj_path=os.path.join(rms_result_visualization_folder, scene_name + '_real_world_predict.obj')
                )
            else:
                predict_polygon_info = from_density_map_result_to_real_world(
                    density_map_rooms= predict_polygon_info,
                    x_range=s3d_bound_info[0],
                    y_range=s3d_bound_info[1],
                    x_min=s3d_bound_info[2],
                    y_min=s3d_bound_info[3],
                )

        las_path = os.path.join(point_cloud_root, scene_name, 'poinc_cloud.las')
        evaluator = RMS_Evaluator(
            las_path=las_path,
            floor_plan_data=predict_polygon_info,
            output_las_result_path=os.path.join(rms_result_visualization_folder, scene_name + '_rms_result.las'),
        )

        rms = evaluator.calculate_rms(mode = 'fast',auto_clip_z=True,need_save_las=save_las_flag)
        print(scene_name + 'RMS: ', rms)
        rms_result_record[scene_name] = rms
        rms_values.append(rms)
    bar.close()
    avg_rms = sum(rms_values) / len(rms_values)
    print('Average RMS: ', avg_rms)
    rms_result_record['avg_rms'] = avg_rms

    with open(rms_result_record_file, 'w') as f:
        json.dump(rms_result_record, f)


def test():
    predict_polygon_info_json = PREDICT_POLYGON_INFO_JSON
    with open(predict_polygon_info_json, 'r') as f:
        predict_polygon_infos = json.load(f)

    adjust_infos = {}
    # print(predict_polygon_infos)
    for info in predict_polygon_infos:
        image_id = info['image_id']
        scene_name = f"scene_{image_id:03d}"
        if adjust_infos.get(scene_name) is None:
            adjust_infos[scene_name] = []
        adjust_infos[scene_name].append(info['segmentation'][0])
    return adjust_infos


if __name__ == '__main__':
    main(is_from_raster=True,is_from_layer0=True,specified_scene_name=['scene_003','scene_004'])

    #test()