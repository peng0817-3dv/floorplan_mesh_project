import os

import hydra
from tqdm import tqdm

from trainer.train_vocabulary import TriangleTokenizationGraphConv,FPTriangleNodes,FPTriangleNodesDataloader
from util.visualization import plot_vertices_and_faces_with_labels

@hydra.main(config_path='../config', config_name='meshgpt', version_base='1.2')
def main(config):
    load_model_experiment_path = ""
    load_checkpoint_name = ""
    checkpoint_path = os.path.join(load_model_experiment_path, "checkpoints", load_checkpoint_name)
    predict_path_root = os.path.join(load_model_experiment_path, "predict")
    if not os.path.exists(predict_path_root):
        os.makedirs(predict_path_root)

    dataset = FPTriangleNodes(config,'test')
    model = TriangleTokenizationGraphConv(config)
    # model = TriangleTokenizationGraphConv.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()
    progress_bar = tqdm(total = len(dataset),desc="Inference")
    for idx,data in dataset:
        # data = dataset.get(0)
        # print(data.batch_size)
        _, _, discretized_vertices, discretized_faces, _ = dataset.get_all_features_for_shape(index=0)
        predict = model.inference_data(data)
        plot_vertices_and_faces_with_labels(vertices=discretized_vertices, faces=discretized_faces,\
                                            labels=predict,output_file=os.path.join(predict_path_root, f"test_{idx}.png"))
        progress_bar.update(idx)

if __name__ == '__main__':
    main()