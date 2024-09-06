import scipy.io as sio
import numpy as np
import torch


def parse_annotations(mat_file_path):
    """
    解析Posetrack2017数据集的.mat标注文件，并提取图像路径、目标框和关键点信息。

    参数:
    mat_file_path -- .mat 文件的路径

    返回:
    images_info -- 包含图像路径、目标框和关键点信息的列表
    """
    # 读取.mat文件
    data = sio.loadmat(mat_file_path)

    # 提取annotations字段
    annolist = data['annolist']
    images_info = []

    # 遍历每一个anno结构
    for anno in annolist[0]:
        # 解析图像路径
        # print(anno['image'])
        image_path = anno['image'][0][0][0][0] if 'image' in anno.dtype.names else None

        # 解析目标框信息
        rects = anno['annorect'] if 'annorect' in anno.dtype.names else None
        annotations = []
        # bboxs = []
        if rects is not None and rects.size > 0:
            for rect in rects[0]:
                # print(len(rect))
                if len(rect) == 0:
                    # print('Invalid rect structure', len(rect))
                    continue
                # 解析目标框坐标
                # bbox={}
                # bbox=[]
                if 'x1' in rect.dtype.names:
                    s = rect['x1']
                    x=1
                    bbox = {
                        'x1': int(rect['x1'][0][0]) ,
                        'y1': int(rect['y1'][0][0]) ,
                        'x2': int(rect['x2'][0][0]) ,
                        'y2': int(rect['y2'][0][0]) ,
                    }
                    bbox = [bbox['x1'],bbox['y1'],bbox['x2'],bbox['y2']]
                    bbox = torch.tensor(bbox)
                    # bboxs.append(bbox)

                    # 解析关键点信息
                    keypoints = []
                    kpt = np.zeros((15, 3), dtype=np.float32)
                    if 'annopoints' in rect.dtype.names:
                        annopoints = rect['annopoints'] if rect['annopoints'].size > 0 else None
                        if annopoints is not None:
                            for annopoint in annopoints[0]:
                                if 'point' in annopoint.dtype.names:
                                    points = annopoint['point'][0]
                                    if points is not None:
                                        for point in points:
                                            # 处理对象类型的数据
                                            id = int(point['id'][0][0])
                                            x = float(point['x'][0][0])
                                            y = float(point['y'][0][0])
                                            is_visible = int(point['is_visible'][0][0])
                                            keypoints.append({
                                                'id': id,
                                                'x': x,
                                                'y': y,
                                                'is_visible': is_visible
                                            })
                                            kpt[id, :] = [x, y, is_visible]
                    kpt = torch.tensor(kpt)
                    annotations.append({
                        'bbox': bbox,
                        # 'keypoints': keypoints,
                        'kpt': kpt
                    })
        # 将信息添加到列表
        if annotations == []:
            # print("annotations == []",image_path)
            continue
        new_annotations=[]

        values = list(zip(*[list(b.values()) for b in annotations]))
        for i, value in enumerate(values):
            value = torch.stack(value, 0)
            new_annotations.append(value)
        new_annotations.append(torch.ones(new_annotations[0].shape[0], dtype=torch.int64))
        images_info.append({
            'image_path': image_path,
            'annotations': new_annotations#[bbox, Keypoints,clas]
            #  bbox[N, 4]    Keypoints, shape[N, 15, 3] and format(x, y, visible) clas [N,]
        })

    return images_info



if __name__ == '__main__':
    # 设置.mat文件的路径
    mat_file_path = r'E:\pc\目标识别相关1\PoseTrack2017\posetrack_data\annotations\train\24487_mpii_relpath_5sec_trainsub.mat'

    # 解析标注文件
    images_info = parse_annotations(mat_file_path)
    # print('...'*40,'\n')
    print(len(images_info))
    # 打印前几个条目的信息
    for info in images_info[:19]:
        # print(info['image_path'])
        # print(len(info['annotations']))
        print(info)
        print(info['annotations'][0].shape)
        print(info['annotations'][1].shape)
        print(info['annotations'][2].shape)
    """
        需要的形式：
        # clas [N,]     bbox [N, 4]    Keypoints, shape [N, 15, 3] and format (x, y, visible).
    """
