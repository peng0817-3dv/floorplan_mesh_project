## MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers

<hr/>

[**arXiv**](http://arxiv.org/abs/2311.15475) | [**Video**](https://www.youtube.com/watch?v=UV90O1_69_o) | [**Project Page**](https://nihalsid.github.io/mesh-gpt/) <br/>


This repository contains the implementation for the paper:

[**MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers**](http://arxiv.org/abs/2311.15475) by Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, Matthias Nießner.

<div>
<div style="text-align: center">
  <img src="https://private-user-images.githubusercontent.com/932110/313438174-05cc7c73-53c7-4d8c-9514-bd2f8a7d7ed0.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA2MzcxMjIsIm5iZiI6MTcxMDYzNjgyMiwicGF0aCI6Ii85MzIxMTAvMzEzNDM4MTc0LTA1Y2M3YzczLTUzYzctNGQ4Yy05NTE0LWJkMmY4YTdkN2VkMC5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMxN1QwMDUzNDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04YjZiYTgyMTU5NzM3MTk4YTYyNTc1Njk2Y2UxZWJjZTRjODkzYzViNDFlMmExYjkzMDBhOTU4YWJmZDJlZTJkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.2uVUaTidnV3_b1V_WfTbsdwZXGzUr3otFp3wqR6tSvI" alt="animated" />
</div>
<div style="margin-top: 5px;">
MeshGPT creates triangle meshes by autoregressively sampling from a transformer model that has been trained to produce tokens from a learned geometric vocabulary. These tokens can then be decoded into the faces of a triangle mesh. Our method generates clean, coherent, and compact meshes, characterized by sharp edges and high fidelity.
</div>
</div>

## Dependencies

Install requirements from the project root directory:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install -r requirements.txt
```
In case errors show up for missing packages, install them manually.

## Structure

Overall code structure is as follows:

| Folder                 | Description                                                                                  |
|------------------------|----------------------------------------------------------------------------------------------|
| `config/`              | hydra  configs                                                                        |
| `data/`                | processed dataset
| `dataset/`             | pytorch datasets and dataloaders                                                           |
| `docs/`                | project webpage files                                                                        |
| `inference/`           | scripts for inferencing trained model                                           |
| `model/`               | pytorch modules for encoder, decoder and the transformer              |
| `pretrained/` | pretrained models on shapenet chairs and tables |
| `runs/`                | model training logs and checkpoints go here in addition to wandb                            |
| `trainer/`             | pytorch-lightning module for training                                                        | 
| `util/`                | misc utilities for positional encoding, visualization, logging etc.                      |


## Training

For launching training, use the following command from project root.
And because we change the code of training for adapt to the structured3d dataset,the code in this branch is not compatible with the original code.
we need to specify the dataset path in the config file(config/meshgpt.yaml) or use the command line arguments.

the "dataset_root" means the directory where the scenes data are stored,like:

```python
dataset_root
--|--scene0000
--|--|--vertices.shp
--|--|--faces.shp
--|--|--edges.shp
--|--scene0001
--|--|--vertices.shp
--|--|--faces.shp
--|--|--edges.shp
--|--...
```

```
# only for structured3d 
python trainer/train_vocabulary.py <options> dataset_root=<path_to_dataset_root>
```

After first loading the dataset, the cache file will be saved in the directory of the dataset root as "cache.pkl".

### Running inference

To run inference use the following command.We need to specify the checkpoint path or specify it in the config file(config/meshgpt.yaml).

You can find the checkpoint path in the "runs/<experiment_name>/checkpoints/<checkpoint_number>.ckpt".
And the result will be saved in the "runs/<experiment_name>/checkpoints/from_checkpoint_<checkpoint_number>/" directory.

```bash
python inference/infer_meshgpt.py  load_checkpoint_path=<ckpt_path> 
```

Examples:

```bash
python inference/infer_meshgpt.py laod_checkpoint_path=runs/transformer_ft_03001627/checkpoints/2287-0.ckpt
```



## License
<a property="dct:title" rel="cc:attributionURL" href="https://nihalsid.github.io/mesh-gpt/">MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://nihalsid.github.io/">Mohd Yawar Nihal Siddiqui</a> is licensed under [Automotive Development Public Non-Commercial License Version 1.0](LICENSE), however portions of the project are available under separate license terms: e.g. NanoGPT code is under MIT license.

## Citation

If you wish to cite us, please use the following BibTeX entry:

```BibTeX
@InProceedings{siddiqui_meshgpt_2024,
    title={MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers},
    author={Siddiqui, Yawar and Alliegro, Antonio and Artemov, Alexey and Tommasi, Tatiana and Sirigatti, Daniele and Rosov, Vladislav and Dai, Angela and Nie{\ss}ner, Matthias},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024},
}

```