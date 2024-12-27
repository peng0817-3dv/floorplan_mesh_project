import os
import random

import matplotlib.pyplot as plt
import numpy as np
from util.s3d_data_load import read_s3d_mesh_info, global_label_colors, enum_label, get_faces_coord
from util.visualization import export_face_to_obj


def analyse_dataset_split(splits: list, label_datas, analyse_results_path = None):
    # s = sum(len(split) for split in splits)
    total = []
    fig,axs = plt.subplots(2,2)
    fig.patch.set_facecolor('lightgrey')
    titles = ['Train', 'Val', 'Test']
    for i,split in enumerate(splits):
        label_count = {l.name: 0 for l in enum_label}
        for scene_id in split:
            labels = label_datas[scene_id]
            for label in labels:
                if label == 31 or label == 32:
                    continue
                label_count[enum_label(label).name] += 1
        labels_size = sum(label_count.values())
        sizes = [v/labels_size for v in label_count.values()]
        colors = [global_label_colors[l.value - 1] for l in list(enum_label)]
        labels_name = [l.name for l in list(enum_label)]

        axs[i // 2, i % 2].pie(sizes, labels=labels_name, colors=colors,autopct='%1.1f%%', startangle=90, textprops={'fontsize': 6})
        axs[i // 2, i % 2].set_title(titles[i])
        total.append(label_count)

    label_count = {l.name: 0 for l in enum_label}
    for ele in total:
        for k,v in ele.items():
            label_count[k] += v

    labels_size = sum(label_count.values())
    sizes = [v / labels_size for v in label_count.values()]
    colors = [global_label_colors[l.value - 1] for l in list(enum_label)]
    labels_name = [l.name for l in list(enum_label)]
    axs[1, 1].pie(sizes, labels=labels_name, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 6})
    axs[1, 1].set_title('Total')
    # plt.show()
    plt.savefig(analyse_results_path,dpi = 800)


def random_split(size, train_ratio, val_ratio):
    n = size
    # n = 10
    train_size = int(n*train_ratio)
    val_size = int(n*val_ratio)
    train_idx = np.random.choice(n, train_size, replace=False)

    remaining_idx = np.setdiff1d(np.arange(n), train_idx)
    val_idx_idx = np.random.choice(len(remaining_idx), val_size, replace=False)
    val_idx = remaining_idx[val_idx_idx]
    test_idx = np.setdiff1d(remaining_idx, val_idx)

    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)
    return [train_idx, val_idx, test_idx]

def shp_to_obj(shp_path, obj_path):
    faces_coords = get_faces_coord(shp_path)
    export_face_to_obj(faces_coords, obj_path)

