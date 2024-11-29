import numpy as np
import torch
from collections import OrderedDict


def get_parameters_from_state_dict(state_dict, filter_key):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith(filter_key):
            new_state_dict[k.replace(filter_key + '.', '')] = state_dict[k]
    return new_state_dict


def accuracy(y_pred, y_true, ignore_label=None, device=None):
    y_pred = y_pred.argmax(dim=-1)

    if ignore_label:
        normalizer = torch.sum(y_true != ignore_label)  # type: ignore
        ignore_mask = torch.where(  # type: ignore
            y_true == ignore_label,
            torch.zeros_like(y_true, device=device),
            torch.ones_like(y_true, device=device)
        ).type(torch.float32)
    else:
        normalizer = y_true.shape[0]
        ignore_mask = torch.ones_like(y_true, device=device).type(torch.float32)
    acc = (y_pred.reshape(-1) == y_true.reshape(-1)).type(torch.float32)  # type: ignore
    acc = torch.sum(acc*ignore_mask.flatten())
    return acc / normalizer


def rmse(y_pred, y_true, num_tokens, ignore_labels=(0, 1, 2)):
    mask = torch.logical_and(y_true != ignore_labels[0], y_pred != ignore_labels[0])
    for i in range(1, len(ignore_labels)):
        mask = torch.logical_and(mask, y_true != ignore_labels[i])
        mask = torch.logical_and(mask, y_pred != ignore_labels[i])
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    vertices_pred = (y_pred - 3) / num_tokens - 0.5
    vertices_true = (y_true - 3) / num_tokens - 0.5
    return torch.sqrt(torch.mean((vertices_pred - vertices_true)**2))


def get_create_shapenet_train_val(path):
    from collections import Counter
    import random

    all_shapes = [x.stem for x in list(path.iterdir())]
    all_categories = sorted(list(set(list(x.split("_")[0] for x in all_shapes))))
    counts_all = Counter()
    for s in all_shapes:
        counts_all[s.split("_")[0]] += 1
    validation = random.sample(all_shapes, int(len(all_shapes) * 0.05))
    counts_val = Counter()
    for s in validation:
        counts_val[s.split("_")[0]] += 1
    for c in all_categories:
        print(c, f"{counts_all[c] / len(all_shapes) * 100:.2f}", f"{counts_val[c] / len(validation) * 100:.2f}")
    train = [x for x in all_shapes if x not in validation]
    Path("val.txt").write_text("\n".join(validation))
    Path("train.txt").write_text("\n".join(train))


def scale_vertices(vertices, x_lims=(0.75, 1.25), y_lims=(0.75, 1.25), z_lims=(0.75, 1.25)):
    # scale x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    vertices = np.stack([vertices[:, 0] * x, vertices[:, 1] * y, vertices[:, 2] * z], axis=-1)
    return vertices


def shift_vertices(vertices, x_lims=(-0.1, 0.1), y_lims=(-0.1, 0.1), z_lims=(-0.075, 0.075)):
    # shift x, y, z
    x = np.random.uniform(low=x_lims[0], high=x_lims[1], size=(1,))
    y = np.random.uniform(low=y_lims[0], high=y_lims[1], size=(1,))
    z = np.random.uniform(low=z_lims[0], high=z_lims[1], size=(1,))
    x = max(min(x, 0.5 - vertices[:, 0].max()), -0.5 - vertices[:, 0].min())
    y = max(min(y, 0.5 - vertices[:, 1].max()), -0.5 - vertices[:, 1].min())
    z = max(min(z, 0.5 - vertices[:, 2].max()), -0.5 - vertices[:, 2].min())
    vertices = np.stack([vertices[:, 0] + x, vertices[:, 1] + y, vertices[:, 2] + z], axis=-1)
    return vertices


def normalize_vertices(vertices):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])  # type: ignore
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    return vertices


def top_p_sampling(logits, p):
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def rotate_vertices(vertices, angle):
    """
    对点进行旋转
    :param vertices: 形状为 (N, 3) 的 tensor，每一行是一个顶点 (x, y, z)
    :param angle: 旋转角度 (90 / 180 / 270)
    :return: 旋转后的顶点 (tensor)
    """
    if angle == 90:
        # rotated_vertices = torch.stack([-vertices[:, 1], vertices[:, 0], vertices[:, 2]], dim=1)  # (-y, x, z)
        rotated_vertices = np.array([-vertices[:, 1], vertices[:, 0], vertices[:, 2]]).T
    elif angle == 180:
        # rotated_vertices = torch.stack([-vertices[:, 0], -vertices[:, 1], vertices[:, 2]], dim=1)  # (-x, -y, z)
        rotated_vertices = np.array([-vertices[:, 0], -vertices[:, 1], vertices[:, 2]]).T
    elif angle == 270:
        # rotated_vertices = torch.stack([vertices[:, 1], -vertices[:, 0], vertices[:, 2]], dim=1)  # (y, -x, z)
        rotated_vertices = np.array([vertices[:, 1], -vertices[:, 0], vertices[:, 2]]).T
    else:
        raise ValueError("不支持的旋转角度。请使用 90, 180, 270 之一。")

    return rotated_vertices


def mirror_vertices(vertices, axis):
    """
    对点进行镜像
    :param vertices: 形状为 (N, 3) 的 tensor，每一行是一个顶点 (x, y)
    :param axis: 镜像轴 ('x' / 'y')
    :return: 镜像后的顶点 (tensor)
    """
    if axis == 'x':
        # mirrored_vertices = torch.stack([vertices[:, 0], -vertices[:, 1], vertices[:, 2]], dim=1)  # (x,-y, z)
        mirrored_vertices = np.array([vertices[:, 0], -vertices[:, 1], vertices[:, 2]]).T
    elif axis == 'y':
        # mirrored_vertices = torch.stack([-vertices[:, 0], vertices[:, 1]], vertices[:, 2],dim=1)  # (-x, y, z)
        mirrored_vertices = np.array([-vertices[:, 0], vertices[:, 1], vertices[:, 2]]).T
    else:
        raise ValueError("不支持的镜像轴。请使用 'x' 或 'y'。")

    return mirrored_vertices


if __name__ == "__main__":
    from pathlib import Path
    # get_create_shapenet_train_val(Path("/cluster/gimli/ysiddiqui/ShapeNetCore.v2.meshlab/"))
    logits_ = torch.FloatTensor([[3, -1, 0.5, 0.1],
                                [0, 9, 0.5, 0.1],
                                [0, 0, 5, 0.1]])
    for i in range(5):
        print(top_p_sampling(logits_, 0.9))
