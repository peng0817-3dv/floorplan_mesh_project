import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import os

import hydra
from tqdm import tqdm
import numpy as np
from util.s3d_data_process import process_vertice_by_op_record
from ablation.only_segment_room_and_wall import FPTriangleWithThreeClsNodes, OnlySegmentRoomAndWall
from util.visualization import plot_vertices_and_faces_with_labels, export_mesh_to_shp


def build_dataset_for_cross(dataset_root, config):
    config.dataset_root = dataset_root
    config.train_ratio = 0.0
    config.val_ratio = 0.0
    config.test_ratio = 1.0

    dataset_name = os.path.basename(dataset_root)
    load_checkpoint_path = config.load_checkpoint_path
    load_checkpoint_root = os.path.dirname(load_checkpoint_path)
    checkpoint_name = os.path.basename(load_checkpoint_path).split(".")[0]
    save_inference_path = os.path.join(load_checkpoint_root, f"inference_from_{checkpoint_name}_for_{dataset_name}")

    config.save_inference_path = save_inference_path

    dataset = FPTriangleWithThreeClsNodes(config, 'test')
    return dataset


def inference(config, load_checkpoint_path, dataset):
    save_inference_path = config.save_inference_path
    # checkpoint_path = os.path.join(load_model_experiment_path, "checkpoints", load_checkpoint_name)
    if save_inference_path is None:
        load_checkpoint_root = os.path.dirname(load_checkpoint_path)
        checkpoint_name = os.path.basename(load_checkpoint_path).split(".")[0]
        save_inference_path = os.path.join(load_checkpoint_root, f"inference_from_{checkpoint_name}")
    predict_path_root = save_inference_path

    print(predict_path_root)
    if not os.path.exists(predict_path_root):
        os.makedirs(predict_path_root)
        print(f"make dir {predict_path_root}")

    # model = TriangleTokenizationGraphConv(config)
    model = OnlySegmentRoomAndWall.load_from_checkpoint(checkpoint_path=load_checkpoint_path)
    model.eval()

    # sample num
    sample_num = 10

    sample_num = min(sample_num, len(dataset))

    samples = np.linspace(0, len(dataset) - 1, sample_num).astype(int)
    progress_bar = tqdm(total=len(samples), desc="Inference")
    for idx in samples:
        data = dataset.get(idx)
        _, targets, vertices, faces, _, op = dataset.get_all_features_for_shape(idx)
        # 训练时，为了方便索引，类别从0开始
        # 画图时，为了方便对照数据集的标签序号，类别从1开始
        targets = targets + 1
        targets = np.where(targets == 2, 31, targets)
        targets = np.where(targets == 3, 32, targets)
        # 预测时，每次推断都自动的把标签从类别0开始转化为了类别1开始
        predict = model.inference_data(data)
        predict = np.where(predict == 2, 31, predict)
        predict = np.where(predict == 3, 32, predict)

        name = dataset.get_name(idx)

        plot_vertices_and_faces_with_labels(vertices=vertices, faces=faces, \
                                            labels=predict,
                                            output_path=os.path.join(predict_path_root, f"{name}_predict.png"))
        plot_vertices_and_faces_with_labels(vertices=vertices, faces=faces, \
                                            labels=targets,
                                            output_path=os.path.join(predict_path_root, f"{name}_ground_truth.png"))
        vertices = process_vertice_by_op_record(reverse_op=op, vertices=vertices)
        export_mesh_to_shp(vertices=vertices, faces=faces, \
                           labels=predict, output_path=os.path.join(predict_path_root, f"{name}_shpfile_predict"))
        progress_bar.update(1)


@hydra.main(config_path='../config', config_name='only_segment_room_and_wall', version_base='1.2')
def main(config):
    load_checkpoint_path = config.load_checkpoint_path
    if config.inference_dataset_path is None:
        print("inference_dataset_path:None")
        dataset = FPTriangleWithThreeClsNodes(config, 'test')
    else:
        print(f"inference_dataset_path:{config.inference_dataset_path}")
        dataset = build_dataset_for_cross(config.inference_dataset_path, config)
    # dataset = build_dataset_for_cross("/mnt/data2/pengyan/augment_stru3d_featured_shp_0-250",config)
    inference(config, load_checkpoint_path, dataset)


if __name__ == '__main__':
    main()