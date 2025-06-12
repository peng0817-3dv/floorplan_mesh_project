import os

from PIL import Image
import numpy as np

from util.s3d_data_process import export_density, export_flip_density


def image_to_numpy(image_path):
    try:
        # 打开图片
        image = Image.open(image_path)
        # 将图片转换为numpy数组
        image_array = np.array(image)
        image_array = image_array.astype(np.float32)
        # 进行归一化操作，将像素值缩放到 [0, 1] 区间
        image_array /= 255.0
        return image_array
    except FileNotFoundError:
        print(f"错误：未找到文件 {image_path}。")
    except Exception as e:
        print(f"错误：发生了未知错误 {e}。")
    return None

def main():
    img_root = r"G:\workspace_plane2DDL\RefCode\RoomFormer\data\stru3d\test"
    out_root = r"G:\workspace_plane2DDL\RefCode\RoomFormer\data\stru3d\test_flip"
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    for img_name in os.listdir(img_root):
        img_path = os.path.join(img_root, img_name)
        img_array = image_to_numpy(img_path)
        img_id = img_name.split(".")[0]
        export_flip_density(img_array, out_root, img_id)

if __name__ == '__main__':
    main()