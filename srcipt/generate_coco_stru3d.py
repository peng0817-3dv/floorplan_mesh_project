import json
import os

import cv2
import numpy as np
from pycocotools import coco
from pycocotools.coco import COCO
from tqdm import tqdm
from shapely.geometry import Polygon
from util.s3d_data_process import read_scale_table, generate_augmented_point_cloud_density_map, export_density
from util.visualization import visualization_seg
from skimage import io

type2id = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

colors_12 = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58230",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd7b4"
]


def is_clockwise(points):
    # points is a list of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0


def resort_corners(corners):
    # re-find the starting point and sort corners clockwisely
    x_y_square_sum = corners[:,0]**2 + corners[:,1]**2
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    ## sort points clockwise
    if not is_clockwise(corners_sorted[:,:2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners


def generate_predict_coco_dict(rooms,bound,curr_img_id):
    min_coord = np.array([bound[0],bound[1]])
    max_coord = np.array([bound[2],bound[3]])

    norm_rooms =[]
    img_res = np.array((256, 256))
    for room in rooms:
        norm_room = []
        for point in room:
            norm_point = np.round((point - min_coord) / (max_coord - min_coord) * img_res)
            norm_room.append(norm_point)
        norm_rooms.append(norm_room)

    coco_annotation_dict_list = []
    curr_instance_id = 0
    for poly_ind, polygon in enumerate(norm_rooms):
        polygon = np.array(polygon)
        poly_shapely = Polygon(polygon)
        area = poly_shapely.area

        # assert area > 10
        # if area < 100:
        if area < 100:
            continue

        rectangle_shapely = poly_shapely.envelope

        coco_seg_poly = []
        poly_sorted = resort_corners(polygon)

        for p in poly_sorted:
            coco_seg_poly += list(p)

        # Slightly wider bounding box
        bb_x, bb_y = rectangle_shapely.exterior.xy
        bb_x = np.unique(bb_x)
        bb_y = np.unique(bb_y)
        bb_x_min = np.maximum(np.min(bb_x), 0)
        bb_y_min = np.maximum(np.min(bb_y), 0)

        bb_x_max = np.minimum(np.max(bb_x), 256 - 1)
        bb_y_max = np.minimum(np.max(bb_y), 256 - 1)

        bb_width = (bb_x_max - bb_x_min)
        bb_height = (bb_y_max - bb_y_min)

        coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]

        coco_annotation_dict = {
            "segmentation": [coco_seg_poly],
            "area": area,
            "iscrowd": 0,
            "image_id": curr_img_id,
            "bbox": coco_bb,
            "category_id": 1,
            "id": curr_instance_id}

        coco_annotation_dict_list.append(coco_annotation_dict)
        curr_instance_id += 1
    return coco_annotation_dict_list



def generate_coco_dict(annos, polygons, curr_instance_id, curr_img_id, ignore_types):
    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])

    coco_annotation_dict_list = []

    for poly_ind, (polygon, poly_type) in enumerate(polygons):
        if poly_type in ignore_types:
            continue

        polygon = junctions[np.array(polygon)]

        poly_shapely = Polygon(polygon)
        area = poly_shapely.area

        # assert area > 10
        # if area < 100:
        if poly_type not in ['door', 'window'] and area < 100:
            continue
        if poly_type in ['door', 'window'] and area < 1:
            continue

        rectangle_shapely = poly_shapely.envelope

        ### here we convert door/window annotation into a single line
        if poly_type in ['door', 'window']:
            assert polygon.shape[0] == 4
            midp_1 = (polygon[0] + polygon[1]) / 2
            midp_2 = (polygon[1] + polygon[2]) / 2
            midp_3 = (polygon[2] + polygon[3]) / 2
            midp_4 = (polygon[3] + polygon[0]) / 2

            dist_1_3 = np.square(midp_1 - midp_3).sum()
            dist_2_4 = np.square(midp_2 - midp_4).sum()
            if dist_1_3 > dist_2_4:
                polygon = np.row_stack([midp_1, midp_3])
            else:
                polygon = np.row_stack([midp_2, midp_4])

        coco_seg_poly = []
        poly_sorted = resort_corners(polygon)

        for p in poly_sorted:
            coco_seg_poly += list(p)

        # Slightly wider bounding box
        bound_pad = 2
        bb_x, bb_y = rectangle_shapely.exterior.xy
        bb_x = np.unique(bb_x)
        bb_y = np.unique(bb_y)
        bb_x_min = np.maximum(np.min(bb_x) - bound_pad, 0)
        bb_y_min = np.maximum(np.min(bb_y) - bound_pad, 0)

        bb_x_max = np.minimum(np.max(bb_x) + bound_pad, 256 - 1)
        bb_y_max = np.minimum(np.max(bb_y) + bound_pad, 256 - 1)

        bb_width = (bb_x_max - bb_x_min)
        bb_height = (bb_y_max - bb_y_min)

        coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]

        coco_annotation_dict = {
            "segmentation": [coco_seg_poly],
            "area": area,
            "iscrowd": 0,
            "image_id": curr_img_id,
            "bbox": coco_bb,
            "category_id": type2id[poly_type],
            "id": curr_instance_id}

        coco_annotation_dict_list.append(coco_annotation_dict)
        curr_instance_id += 1

    return coco_annotation_dict_list


def parse_coco_dict(json_path, img_path, num_scenes):

    coco = COCO(json_path)
    catIds = coco.getCatIds()  # 获取指定类别 id
    exclude_cat_ids = [17,18]  # 替换为你要排除的类别 ID
    # 使用列表推导式获取排除指定 ID 之外的所有类别 ID
    include_cat_ids = [cat_id for cat_id in catIds if cat_id not in exclude_cat_ids]

    imgIds = coco.getImgIds()  # 获取图片i

    img = coco.loadImgs(num_scenes)[0]  # 加载图片,loadImgs() 返回的是只有一个内嵌字典元素的list, 使用[0]来访问这个元素
    image = io.imread(os.path.join(img_path,img['file_name']))
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=include_cat_ids, iscrowd=None)
    anns = coco.loadAnns(annIds)
    gt_polygons_list = []
    for ann in anns:
        gt_polygons_list.append(ann['segmentation'][0])

    gt_polygons_list = [np.array(poly).reshape(-1, 2).astype(np.int32) for poly in gt_polygons_list]
    return image, gt_polygons_list



def main():
    out_folder = r'G:\workspace_plane2DDL\augment_point_cloud_density'
    point_cloud_folder = r'I:\s3dParseT5'
    annotation_folder = r'G:\workspace_plane2DDL\data_anno'
    scale_table_path = r'G:\workspace_plane2DDL\data_anno_scales\scales.txt'
    scale_table = read_scale_table(scale_table_path)

    data_parts = os.listdir(point_cloud_folder)
    output_annotation_folder = os.path.join(out_folder, 'annotations')
    ### prepare

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    if not os.path.exists(output_annotation_folder):
        os.mkdir(output_annotation_folder)

    train_img_folder = os.path.join(out_folder, 'train')
    val_img_folder = os.path.join(out_folder, 'val')
    test_img_folder = os.path.join(out_folder, 'test')

    for img_folder in [train_img_folder, val_img_folder, test_img_folder]:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

    coco_train_json_path = os.path.join(output_annotation_folder, 'train.json')
    coco_val_json_path = os.path.join(output_annotation_folder, 'val.json')
    coco_test_json_path = os.path.join(output_annotation_folder, 'test.json')

    coco_train_dict = {"images":[],"annotations":[],"categories":[]}
    coco_val_dict = {"images":[],"annotations":[],"categories":[]}
    coco_test_dict = {"images":[],"annotations":[],"categories":[]}

    for key, value in type2id.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        coco_train_dict["categories"].append(type_dict)
        coco_val_dict["categories"].append(type_dict)
        coco_test_dict["categories"].append(type_dict)

    ### begin processing
    instance_id = 0

    for scene in tqdm(data_parts):
        scene_path = os.path.join(point_cloud_folder, scene)
        annotation_path = os.path.join(annotation_folder, scene, 'annotation_3d.json')
        scene_id = scene.split('_')[-1]
        val = scale_table.get(scene)
        if not scale_table.get(scene):
            continue
        las_path = os.path.join(scene_path, 'scale.laz')
        if not os.path.exists(las_path):
            continue
        rooms,na,density = generate_augmented_point_cloud_density_map(scene_path,
                                                   annotation_path,
                                                   scale_table)


        ### prepare coco dict
        img_id = int(scene_id)
        img_dict = {}
        img_dict["file_name"] = scene_id + '.png'
        img_dict["id"] = img_id
        img_dict["width"] = 256
        img_dict["height"] = 256
        polygons_list = generate_coco_dict(na, rooms, instance_id, img_id, ignore_types=['outwall'])
        instance_id += len(polygons_list)

        ### train
        if int(scene_id) < 3000:
            coco_train_dict["images"].append(img_dict)
            coco_train_dict["annotations"] += polygons_list
            export_density(density, train_img_folder, scene_id)

        ### val
        elif int(scene_id) >= 3000 and int(scene_id) < 3250:
            coco_val_dict["images"].append(img_dict)
            coco_val_dict["annotations"] += polygons_list
            export_density(density, val_img_folder, scene_id)

        ### test
        else:
            coco_test_dict["images"].append(img_dict)
            coco_test_dict["annotations"] += polygons_list
            export_density(density, test_img_folder, scene_id)

        print(scene_id)


    with open(coco_train_json_path, 'w') as f:
        json.dump(coco_train_dict, f)
    with open(coco_val_json_path, 'w') as f:
        json.dump(coco_val_dict, f)
    with open(coco_test_json_path, 'w') as f:
        json.dump(coco_test_dict, f)

def visualize_result():
    out_folder = r'G:\workspace_plane2DDL\augment_point_cloud_density'
    img_folder = os.path.join(out_folder, 'train')
    annotation_json_path = os.path.join(out_folder, 'annotations', 'train.json')
    visualization_seg(20, annotation_json_path, img_folder)#2104

if __name__ == '__main__':
    main()


