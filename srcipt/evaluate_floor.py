import os

from pycocotools import coco

from srcipt.generate_coco_stru3d import parse_coco_dict
from util.Evaluator import Evaluator
from util.merge_polygon import MergePolygonSolution

density_folder = r'G:\workspace_plane2DDL\augment_point_cloud_density'
img_folder = os.path.join(density_folder, 'test')
annotation_json_path = os.path.join(density_folder, 'annotations', 'test.json')

def evaluate_single_floor(predict_floor:dict,):
    '''
    predict_floor: dict, {'vertex': (n_v, 3), 'faces': (n_f, 3), 'floor_label': (n_f, 1)}
    '''
    pass

def evaluate_floor_with_mockdata():
    mock_data_path = r'G:\workspace_plane2DDL\testData\10_percent_box\scene_00020'

    merged_solution = MergePolygonSolution()
    merged_solution.load_data_from_shp_file(mock_data_path)
    merged_solution.start_work()
    predict_rooms = merged_solution.get_merged_polygons_as_raster_format()

    scene_id = '00020'
    scene_num = int(scene_id)

    mock_img_folder = os.path.join(density_folder, 'train')
    mock_json_path = os.path.join(density_folder, 'annotations', 'train.json')
    density,gt_polygons_list = parse_coco_dict(json_path=mock_json_path, img_path=mock_img_folder, num_scenes=scene_num)
    gt_data = {
        'density':density,
        'polygons_list':gt_polygons_list
    }
    cur_scene_eval = Evaluator(gt_data)
    quan_result_dict =cur_scene_eval.evaluate_scene(predict_rooms)
    print(quan_result_dict)

if __name__ == '__main__':
    evaluate_floor_with_mockdata()