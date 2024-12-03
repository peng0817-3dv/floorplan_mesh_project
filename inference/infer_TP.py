import os

import hydra
from tqdm import tqdm

from trainer.train_vocabulary import TriangleTokenizationGraphConv,FPTriangleNodes,FPTriangleNodesDataloader
from util.visualization import plot_vertices_and_faces_with_labels

@hydra.main(config_path='../config', config_name='meshgpt', version_base='1.2')
def main(config):
    load_checkpoint_path = config.load_checkpoint_path
    save_inference_path = config.save_inference_path
    # checkpoint_path = os.path.join(load_model_experiment_path, "checkpoints", load_checkpoint_name)
    if save_inference_path is None:
        load_checkpoint_root = os.path.dirname(load_checkpoint_path)
        checkpoint_name = os.path.basename(load_checkpoint_path).split(".")[0]
        save_inference_path = os.path.join(load_checkpoint_root, f"inference_from_{checkpoint_name}")
    predict_path_root = save_inference_path
    if not os.path.exists(predict_path_root):
        os.makedirs(predict_path_root)

    dataset = FPTriangleNodes(config,'test')
    # model = TriangleTokenizationGraphConv(config)
    model = TriangleTokenizationGraphConv.load_from_checkpoint(checkpoint_path=load_checkpoint_path)
    model.eval()
    progress_bar = tqdm(total = len(dataset),desc="Inference")
    for idx in range(len(dataset)):
        data = dataset.get(idx)
        # print(data.batch_size)
        _, _, vertices, faces, _ = dataset.get_all_features_for_shape(idx)
        predict = model.inference_data(data)
        if idx % 10 == 0:
            plot_vertices_and_faces_with_labels(vertices=vertices, faces=faces,\
                                            labels=predict,output_file=os.path.join(predict_path_root, f"test_{idx}.png"))
        progress_bar.update(1)

if __name__ == '__main__':
    main()