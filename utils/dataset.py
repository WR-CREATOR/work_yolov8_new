import numpy as np
import os
import pandas as pd
import random
import scipy.io
import torch
import torchvision.transforms as T
from Cython.Compiler.Future import annotations
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils.parse_annotation import parse_annotations


class PoseTrackDataset(Dataset):

    def __init__(self, annotations_paths='./PoseTrack2017/posetrack_data/annotations/train', n_sequences = 1, temporal = 5, n_joints = 15, std = 1):
        # Model input size
        self.height = 640
        self.width = 640
        # Length of temporal sequence
        self.temporal = temporal
        # The number of joints that we want to predict their position
        self.n_joints = n_joints
        # sigma of joint position maps
        self.std = std
        # Generate temporal sequences
        self.temporal_sequences = self.gen_temporal_seq(annotations_paths,  temporal, n_sequences)

    def gen_temporal_seq(self, annotations_paths, temporal, n_sequences):
        """
        Generate temporal sequences
        Args:
            annotations_paths:
            temporal: 生成几个视频帧数
            n_sequences: 一个视频生成几个视频

        Returns:
            temporal_sequences:
        """
        files = os.listdir(annotations_paths)
        temporal_sequences = []
        for i,annotation in enumerate(files):
            if annotation.split('.')[-1] == 'mat':
                annotation_path=os.path.join(annotations_paths, annotation)
                frames_info = parse_annotations(annotation_path)
                n_frames = len(frames_info)
                # Ignore videos with frames less than the temporal seq length
                if n_frames < self.temporal: continue

                for _ in range(n_sequences):
                    seq = []
                    start_index = random.randint(0, n_frames - temporal)

                    for k in range(start_index, (start_index + temporal)):
                        seq.append([frames_info[k],k])

                    temporal_sequences.append(seq)
        return temporal_sequences # [seq,...]:seq->[[frames_info,frames_index],...]

    def __getitem__(self, item):
        # Load the frames (.jpg) of sequence
        frames = self.temporal_sequences[item]

        # ( images = model input) shape : (t*3) * 640 * 640
        images = torch.zeros(self.temporal * 3, self.width, self.height)
        bboxs, kpts, clas,temporal_idx=[],[],[],[]

        for i in range(self.temporal):
            img_path = frames[i][0]['image_path']
            annotations = frames[i][0]['annotations']
            img = Image.open('E:\pc\目标识别相关1\PoseTrack2017\posetrack_data/'+img_path)
            h, w, c = np.array(img).shape
            # Get the ratio between raw image size and target size
            # In the following we need these to correction joints position after image resizing
            ratio_x = self.width / float(w)
            ratio_y = self.height / float(h)
            # normalize image
            img_transformer = self.get_img_transformer(self.width, self.height)
            img = img_transformer(img)
            images[(i * 3): (i * 3 + 3), :, :] = img
            # 由于需要temporal个图片合在一起，图片中检测数量不一样，所以需要temporal_idx来区分
            # 检测框，关键点，类别
            bboxs_, kpts_, clas_ = annotations[0], annotations[1], annotations[2]
            temporal_idx_  = torch.zeros(len(bboxs_))
            print(len(bboxs_))
            temporal_idx_ += i  # add target image index for build_targets()
            bboxs.append(bboxs_)
            kpts.append(kpts_)
            clas.append(clas_)
            temporal_idx.append(temporal_idx_)
        bboxs = torch.cat(bboxs,0)
        kpts = torch.cat(kpts,0)
        clas = torch.cat(clas,0)
        temporal_idx = torch.cat(temporal_idx,0)
        batch_idx = torch.zeros(len(bboxs))

        return {'img':images.float(),'cls':clas,'keypoints':kpts,'bboxes':bboxs,'temporal_idx':temporal_idx,'batch_idx':batch_idx}
        # 单个
        # clas [N,]     bbox [N, 4]    Keypoints, shape [N, 15, 3] and format (x, y, visible).
        # batch_idx torch.zeros(len(bbox))
        # bbox_area ratio_pad resized_shape ori_shape

        # 多个 这样不行，目标数量不一样
        # images [temporal * 3,h,w]  clas[[temporal,m,1]] bbox [temporal,m,5] kpts [temporal,m,15,3]
        # 需要cat起来使用temporal_idx进行索引

    def get_img_transformer(self, width, height):
        return T.Compose([
            T.Resize((width, height), interpolation = T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.temporal_sequences)

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        """
        [
            (batch1["image"], batch2["image"]),
            (batch1["target"], batch2["target"]),
            (batch1["batch_idx"], batch2["batch_idx"])
        ]
        """
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb",'temporal_idx'}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"]) # [batch1["batch_idx"], batch2["batch_idx"],...]
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
        # 一个样本有一个对应的batch_idx，0，1，2... 叫image index更合适


def load_dataset(data_path):
    frames_dir = '/frames/'
    clips = os.listdir(data_path + frames_dir)
    random.shuffle(clips)

    train_frames = [data_path + frames_dir + str(x) for x in clips[:1258]]
    test_frames = [data_path + frames_dir + str(x) for x in clips[1258:]]

    train_labels = [(str(x) + '.mat').replace('frames', 'labels') for x in train_frames]
    test_labels = [(str(x) + '.mat').replace('frames', 'labels') for x in test_frames]

    print('-' * 40)
    print('Train set  - total number of videos =', len(train_labels))
    print('Test set - total number of videos = ', len(test_labels))

    return train_frames, train_labels, test_frames, test_labels


def get_data_loaders(train_annotations_paths, val_annotations_paths, train_bs, val_bs):
    train_data = PoseTrackDataset(train_annotations_paths)
    val_data = PoseTrackDataset(val_annotations_paths)

    print('-' * 40)
    print('Train samples ( sample = a sequence of frames) =', len(train_data))
    print('Validation samples =', len(val_data))

    train_dl = DataLoader(train_data, batch_size = train_bs, shuffle = True)
    val_dl = DataLoader(val_data, batch_size = val_bs, shuffle = True)

    return train_dl, val_dl

if __name__ == '__main__':
    posetrackdataset = PoseTrackDataset(annotations_paths=r'E:\pc\目标识别相关1\PoseTrack2017\posetrack_data\annotations\train',temporal = 5)
    s = posetrackdataset[0]
    # print(s)
    print('-'*50)
    train_dl = DataLoader(posetrackdataset, batch_size=8, collate_fn=posetrackdataset.collate_fn,shuffle=False)
    for i, batch in enumerate(train_dl):
        # print(batch)
        print(batch['img'].shape)
        print(batch['keypoints'].shape)
        print(batch['bboxes'].shape)
        print(batch['temporal_idx'].shape)
        print(batch['batch_idx'].shape)
        print(batch['cls'].shape)
        """
        torch.Size([8, 15, 640, 640])
        torch.Size([210, 15, 3])
        torch.Size([210, 4])
        torch.Size([210])
        torch.Size([210])
        torch.Size([210])
        """
        print('-' * 50)
        print(batch['temporal_idx'])
        print(batch['batch_idx'])
        print(batch['keypoints'])
        break

    # images= posetrackdataset[0]
    # print(images['img'].shape)
    # print(posetrackdataset[0])
    # 多目标输入是什么？(b,m,5+ktp_shape)
    # 通过loss得知，传入的batch是字典?
