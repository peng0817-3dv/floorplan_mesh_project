## xxx

<hr/>

[**arXiv**]()  <br/>


This repository contains the implementation for the paper:

[**xxx**]() by xxx

[//]: # (<div>)

[//]: # (<div style="text-align: center">)

[//]: # (  <img src="https://private-user-images.githubusercontent.com/932110/313438174-05cc7c73-53c7-4d8c-9514-bd2f8a7d7ed0.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTA2MzcxMjIsIm5iZiI6MTcxMDYzNjgyMiwicGF0aCI6Ii85MzIxMTAvMzEzNDM4MTc0LTA1Y2M3YzczLTUzYzctNGQ4Yy05NTE0LWJkMmY4YTdkN2VkMC5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwMzE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDMxN1QwMDUzNDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04YjZiYTgyMTU5NzM3MTk4YTYyNTc1Njk2Y2UxZWJjZTRjODkzYzViNDFlMmExYjkzMDBhOTU4YWJmZDJlZTJkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.2uVUaTidnV3_b1V_WfTbsdwZXGzUr3otFp3wqR6tSvI" alt="animated" />)

[//]: # (</div>)

[//]: # (<div style="margin-top: 5px;">)

[//]: # (MeshGPT creates triangle meshes by autoregressively sampling from a transformer model that has been trained to produce tokens from a learned geometric vocabulary. These tokens can then be decoded into the faces of a triangle mesh. Our method generates clean, coherent, and compact meshes, characterized by sharp edges and high fidelity.)

[//]: # (</div>)

[//]: # (</div>)

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

| Folder        | Description                                                         |
|---------------|---------------------------------------------------------------------|
| `config/`     | hydra  configs                                                      |
| `data/`       | processed dataset                                                   
| `dataset/`    | pytorch datasets and dataloaders                                    |
| `docs/`       | project webpage files                                               |
| `inference/`  | scripts for inferencing trained model                               |
| `model/`      | pytorch modules for encoder, decoder and the transformer            |
| `pretrained/` | pretrained models on shapenet chairs and tables                     |
| `runs/`       | model training logs and checkpoints go here in addition to wandb    |
| `trainer/`    | pytorch-lightning module for training                               | 
| `util/`       | misc utilities for positional encoding, visualization, logging etc. |
| `script/`     | some scripts for visualization,data process,et.                     |


## Training

For launching training, use the following command from project root.
And because we change the code of training for adapt to the structured3d dataset,the code in this branch is not compatible with the original code.
we need to specify the dataset path in the config file(config/graph_transformer.yaml) or use the command line arguments.

we provide our dataset in this [link](https://drive.google.com/file/d/1bafOvmRw7Q0DWuvvb14nVipXvUNbw4UL/view?usp=sharing).you can download and unzip it to the dataset root directory.


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
python trainer/train_triangle.py <options> dataset_root=<path_to_dataset_root>
```

After first loading the dataset, the cache file will be saved in the directory of the dataset root as "cache.pkl".

### Running inference

To run inference use the following command.We need to specify the checkpoint path or specify it in the config file(config/meshgpt.yaml).

You can find the checkpoint path in the "runs/<experiment_name>/checkpoints/<checkpoint_number>.ckpt".
And the result will be saved in the "runs/<experiment_name>/checkpoints/from_checkpoint_<checkpoint_number>/" directory.

Also we provide a pre-trained model checkpoint in this [link](https://drive.google.com/file/d/1bafOvmRw7Q0DWuvvb14nVipXvUNbw4UL/view?usp=sharing).


Best practice is to specify the inference dataset path,and if not, it will use the dataset_path which in the yaml file, and it will use the dataset test part to infer.

You can use the s3d pre-trained model to infer our provide structured3d dataset,and the dataset GT annotation file can be download from this [link](https://polybox.ethz.ch/index.php/s/wKYWFsQOXHnkwcG).

You can also use the s3d pre-trained model to infer other dataset.Your own dataset should follow our process style in ArcGis shp files.And the GT annotation shuold 
follow the style in COCO format which like [RoomFormer](https://github.com/ywyue/RoomFormer) procject.
we provide sample dataset in this [link](https://drive.google.com/file/d/1GoHVpp4-BqibvR1mDtQos8yKmGGbt2mP/view?usp=sharing),and the related gt annotation file in this [link](https://drive.google.com/file/d/1cl6GvA2Fv7LgAz5RKA70M4O255y_qxFN/view?usp=sharing).

```bash
python inference/eval_triangle.py  load_checkpoint_path=<ckpt_path>
                                   <options>inference_dataset_path = <path_to_inference_data>
                                   gt_coco_path = <gt_coco_path> 
```






## License


## Citation

If you wish to cite us, please use the following BibTeX entry:

```BibTeX


```