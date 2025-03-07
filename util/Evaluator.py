import cv2
import numpy as np

corner_metric_thresh = 10
angle_metric_thresh = 5

class Evaluator:
    def __init__(self,data_rw):
        self.gt_data = data_rw

    def evaluate_scene(self, room_polys):
        # Ground Truth

        gt_polys_list = self.gt_data["polygons_list"]
        # 闭合处理
        gt_polys_list = [np.concatenate([poly, poly[None, 0]]) for poly in gt_polys_list]
        # 闭合处理
        room_polys = [np.concatenate([poly, poly[None, 0]]) for poly in room_polys]

        img_size = (256, 256)
        quant_result_dict = self.get_quantitative(
                                    gt_polys= gt_polys_list,
                                    gt_polys_types = None,
                                    gt_window_doors = None,
                                    gt_window_doors_types = None,
                                    ignore_mask_region = None,
                                    pred_polys=room_polys,
                                    pred_types = None,
                                    pred_window_doors = None,
                                    pred_window_doors_types = None,
                                    masks_list=None,
                                    img_size=img_size,
                                    dataset_type='s3d')
        return quant_result_dict

    def get_quantitative(self, gt_polys, gt_polys_types, gt_window_doors, gt_window_doors_types, ignore_mask_region,
                         pred_polys=None, pred_types=None, pred_window_doors=None, pred_window_doors_types=None,
                         masks_list=None, img_size=(256, 256), dataset_type="s3d"):

        def get_room_metric():
            pred_overlaps = [False] * len(pred_room_map_list)

            for pred_ind1 in range(len(pred_room_map_list) - 1):
                pred_map1 = pred_room_map_list[pred_ind1]

                for pred_ind2 in range(pred_ind1 + 1, len(pred_room_map_list)):
                    pred_map2 = pred_room_map_list[pred_ind2]

                    if dataset_type == "s3d":
                        kernel = np.ones((5, 5), np.uint8)
                    else:
                        kernel = np.ones((3, 3), np.uint8)

                    # todo: for our method, the rooms share corners and edges, need to check here
                    pred_map1_er = cv2.erode(pred_map1, kernel)
                    pred_map2_er = cv2.erode(pred_map2, kernel)

                    intersection = (pred_map1_er + pred_map2_er) == 2
                    # intersection = (pred_map1 + pred_map2) == 2

                    intersection_area = np.sum(intersection)

                    if intersection_area >= 1:
                        pred_overlaps[pred_ind1] = True
                        pred_overlaps[pred_ind2] = True

            # import pdb; pdb.set_trace()
            room_metric = [np.bool_((1 - pred_overlaps[ind]) * pred2gt_exists[ind]) for ind in range(len(pred_polys))]
            room_sem_metric = None
            return room_metric, room_sem_metric

        def get_corner_metric():

            room_corners_metric = []
            for pred_poly_ind, gt_poly_ind in enumerate(pred2gt_indices):
                p_poly = pred_polys[pred_poly_ind][:-1] # Last vertex = First vertex

                p_poly_corner_metrics = [False] * p_poly.shape[0]
                if not room_metric[pred_poly_ind]:
                    room_corners_metric += p_poly_corner_metrics
                    continue

                gt_poly = gt_polys[gt_poly_ind][:-1]

                # for v in p_poly:
                #     v_dists = np.linalg.norm(v[None,:] - gt_poly, axis=1, ord=2)
                #     v_min_dist = np.min(v_dists)
                #
                #     v_tp = v_min_dist <= 10
                #     room_corners_metric.append(v_tp)

                for v in gt_poly:
                    v_dists = np.linalg.norm(v[None,:] - p_poly, axis=1, ord=2)
                    v_min_dist_ind = np.argmin(v_dists)
                    v_min_dist = v_dists[v_min_dist_ind]

                    if not p_poly_corner_metrics[v_min_dist_ind]:
                        v_tp = v_min_dist <= corner_metric_thresh
                        p_poly_corner_metrics[v_min_dist_ind] = v_tp

                room_corners_metric += p_poly_corner_metrics

            return room_corners_metric

        def get_angle_metric():

            def get_line_vector(p1, p2):
                p1 = np.concatenate((p1, np.array([1])))
                p2 = np.concatenate((p2, np.array([1])))

                line_vector = -np.cross(p1, p2)

                return line_vector

            def get_poly_orientation(my_poly):
                angles_sum = 0
                for v_ind, _ in enumerate(my_poly):
                    if v_ind < len(my_poly) - 1:
                        v_sides = my_poly[[v_ind - 1, v_ind, v_ind, v_ind + 1], :]
                    else:
                        v_sides = my_poly[[v_ind - 1, v_ind, v_ind, 0], :]

                    v1_vector = get_line_vector(v_sides[0], v_sides[1])
                    v1_vector = v1_vector / (np.linalg.norm(v1_vector, ord=2) + 1e-4)
                    v2_vector = get_line_vector(v_sides[2], v_sides[3])
                    v2_vector = v2_vector / (np.linalg.norm(v2_vector, ord=2) + 1e-4)

                    orientation = (v_sides[1, 1] - v_sides[0, 1]) * (v_sides[3, 0] - v_sides[1, 0]) - (
                            v_sides[3, 1] - v_sides[1, 1]) * (
                                          v_sides[1, 0] - v_sides[0, 0])

                    v1_vector_2d = v1_vector[:2] / (v1_vector[2] + 1e-4)
                    v2_vector_2d = v2_vector[:2] / (v2_vector[2] + 1e-4)

                    v1_vector_2d = v1_vector_2d / (np.linalg.norm(v1_vector_2d, ord=2) + 1e-4)
                    v2_vector_2d = v2_vector_2d / (np.linalg.norm(v2_vector_2d, ord=2) + 1e-4)

                    angle_cos = v1_vector_2d.dot(v2_vector_2d)
                    angle_cos = np.clip(angle_cos, -1, 1)

                    # G.T. has clockwise orientation, remove minus in the equation

                    angle = np.sign(orientation) * np.abs(np.arccos(angle_cos))
                    angle_degree = angle * 180 / np.pi

                    angles_sum += angle_degree

                return np.sign(angles_sum)

            def get_angle_v_sides(inp_v_sides, poly_orient):
                v1_vector = get_line_vector(inp_v_sides[0], inp_v_sides[1])
                v1_vector = v1_vector / (np.linalg.norm(v1_vector, ord=2) + 1e-4)
                v2_vector = get_line_vector(inp_v_sides[2], inp_v_sides[3])
                v2_vector = v2_vector / (np.linalg.norm(v2_vector, ord=2) + 1e-4)

                orientation = (inp_v_sides[1, 1] - inp_v_sides[0, 1]) * (inp_v_sides[3, 0] - inp_v_sides[1, 0]) - (
                        inp_v_sides[3, 1] - inp_v_sides[1, 1]) * (
                                      inp_v_sides[1, 0] - inp_v_sides[0, 0])

                v1_vector_2d = v1_vector[:2] / (v1_vector[2]+ 1e-4)
                v2_vector_2d = v2_vector[:2] / (v2_vector[2]+ 1e-4)

                v1_vector_2d = v1_vector_2d / (np.linalg.norm(v1_vector_2d, ord=2) + 1e-4)
                v2_vector_2d = v2_vector_2d / (np.linalg.norm(v2_vector_2d, ord=2) + 1e-4)

                angle_cos = v1_vector_2d.dot(v2_vector_2d)
                angle_cos = np.clip(angle_cos, -1, 1)

                angle = poly_orient * np.sign(orientation) * np.arccos(angle_cos)
                angle_degree = angle * 180 / np.pi

                return angle_degree

            room_angles_metric = []
            for pred_poly_ind, gt_poly_ind in enumerate(pred2gt_indices):
                p_poly = pred_polys[pred_poly_ind][:-1] # Last vertex = First vertex

                p_poly_angle_metrics = [False] * p_poly.shape[0]
                if not room_metric[pred_poly_ind]:
                    room_angles_metric += p_poly_angle_metrics
                    continue

                gt_poly = gt_polys[gt_poly_ind][:-1]

                # for v in p_poly:
                #     v_dists = np.linalg.norm(v[None,:] - gt_poly, axis=1, ord=2)
                #     v_min_dist = np.min(v_dists)
                #
                #     v_tp = v_min_dist <= 10
                #     room_corners_metric.append(v_tp)

                gt_poly_orient = get_poly_orientation(gt_poly)
                p_poly_orient = get_poly_orientation(p_poly)

                for v_gt_ind, v in enumerate(gt_poly):
                    v_dists = np.linalg.norm(v[None,:] - p_poly, axis=1, ord=2)
                    v_ind = np.argmin(v_dists)
                    v_min_dist = v_dists[v_ind]

                    if v_min_dist > corner_metric_thresh:
                        # room_angles_metric.append(False)
                        continue

                    if v_ind < len(p_poly) - 1:
                        v_sides = p_poly[[v_ind - 1, v_ind, v_ind, v_ind + 1], :]
                    else:
                        v_sides = p_poly[[v_ind - 1, v_ind, v_ind, 0], :]

                    v_sides = v_sides.reshape((4,2))
                    pred_angle_degree = get_angle_v_sides(v_sides, p_poly_orient)

                    # Note: replacing some variables with values from the g.t. poly

                    if v_gt_ind < len(gt_poly) - 1:
                        v_sides = gt_poly[[v_gt_ind - 1, v_gt_ind, v_gt_ind, v_gt_ind + 1], :]
                    else:
                        v_sides = gt_poly[[v_gt_ind - 1, v_gt_ind, v_gt_ind, 0], :]

                    v_sides = v_sides.reshape((4, 2))
                    gt_angle_degree = get_angle_v_sides(v_sides, gt_poly_orient)

                    angle_metric = np.abs(pred_angle_degree - gt_angle_degree)

                    # room_angles_metric.append(angle_metric < 5)
                    p_poly_angle_metrics[v_ind] = angle_metric <= angle_metric_thresh

                    # if angle_metric > 5:
                    #     print(v_gt_ind, angle_metric)
                    #     print(pred_angle_degree, gt_angle_degree)
                    #     input("?")


                room_angles_metric += p_poly_angle_metrics

            for am, cm in zip(room_angles_metric, corner_metric):
                assert not (cm == False and am == True), "cm: %d am: %d" %(cm, am)

            return room_angles_metric

        h, w = img_size

        gt_room_map_list = []
        for room_ind, poly in enumerate(gt_polys):
            room_map = np.zeros((h, w))
            cv2.fillPoly(room_map, [poly], color=1.)
            gt_room_map_list.append(room_map)

        def poly_map_sort_key(x):
            return np.sum(x[1])

        gt_polys_sorted_indcs = [i[0] for i in sorted(enumerate(gt_room_map_list), key=poly_map_sort_key, reverse=True)]
        gt_polys = [gt_polys[ind] for ind in gt_polys_sorted_indcs]
        gt_room_map_list = [gt_room_map_list[ind] for ind in gt_polys_sorted_indcs]

        if pred_polys is not None:
            pred_room_map_list = []
            for room_ind, poly in enumerate(pred_polys):
                room_map = np.zeros((h, w))
                cv2.fillPoly(room_map, [poly], color=1.)
                pred_room_map_list.append(room_map)


        gt2pred_indices = [-1] * len(gt_polys)
        gt2pred_exists = [False] * len(gt_polys)


        ### match predicted rooms to ground truth rooms
        for gt_ind, gt_map in enumerate(gt_room_map_list):

            best_iou = 0.
            best_ind = -1

            for pred_ind, pred_map in enumerate(pred_room_map_list):

                intersection = (pred_map + gt_map) == 2
                union =  (pred_map + gt_map) >= 1
                # intersection = (pred_map + gt_map) == 2
                # union = (pred_map + gt_map) >= 1

                iou = np.sum(intersection) / (np.sum(union) + 1)

                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_ind = pred_ind

                    if pred_types is not None:
                        if gt_polys_types[gt_ind] == pred_types[pred_ind]:
                            best_ind_sem = pred_ind

            gt2pred_indices[gt_ind] = best_ind
            gt2pred_exists[gt_ind] = best_ind != -1

        pred2gt_exists = [True if pred_ind in gt2pred_indices else False for pred_ind, _ in enumerate(pred_polys)]
        pred2gt_indices = [gt2pred_indices.index(pred_ind) if pred_ind in gt2pred_indices else -1 for pred_ind, _ in enumerate(pred_polys)]

        room_metric, room_sem_metric = get_room_metric()

        ###### metric for room WITHOUT considering type ######
        if len(pred_polys) == 0:
            room_metric_prec = 0
        else:
            room_metric_prec = sum(room_metric) / float(len(pred_polys))
        room_metric_rec = sum(room_metric) / float(len(gt_polys))

        ###### metric for corner ######
        corner_metric = get_corner_metric()
        pred_corners_n = sum([poly.shape[0] - 1 for poly in pred_polys])
        gt_corners_n = sum([poly.shape[0] - 1 for poly in gt_polys])

        if pred_corners_n > 0:
            corner_metric_prec = sum(corner_metric) / float(pred_corners_n)
        else:
            corner_metric_prec = 0
        corner_metric_rec = sum(corner_metric) / float(gt_corners_n)

        ###### metric for angle ######
        angles_metric = get_angle_metric()

        if pred_corners_n > 0:
            angles_metric_prec = sum(angles_metric) / float(pred_corners_n)
        else:
            angles_metric_prec = 0
        angles_metric_rec = sum(angles_metric) / float(gt_corners_n)


        # sanity check
        assert room_metric_prec <= 1
        assert room_metric_rec <= 1
        assert corner_metric_prec <= 1
        assert corner_metric_rec <= 1
        assert angles_metric_prec <= 1
        assert angles_metric_rec <= 1

        result_dict = {
            'room_prec': room_metric_prec,
            'room_rec': room_metric_rec,
            'corner_prec': corner_metric_prec,
            'corner_rec': corner_metric_rec,
            'angles_prec': angles_metric_prec,
            'angles_rec': angles_metric_rec,
        }

        return result_dict
