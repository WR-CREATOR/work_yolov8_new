import torch

# 假设 batch_size 为 3，每个图像有 17 个关键点
batch_size = 3
num_keypoints = 17
channels = 3
height = 256
width = 256
num_targets = 2  # 每个图像有 2 个目标

batch = {
    "image": torch.randn(batch_size, channels, height, width),
    "keypoints": torch.tensor([
        [
            [10, 20, 1],
            [30, 40, 1],
            [50, 60, 1],
            # ... (17 keypoints)
        ],
        [
            [20, 30, 1],
            [40, 50, 1],
            [60, 70, 1],
            # ... (17 keypoints)
        ],
        [
            [30, 40, 1],
            [50, 60, 1],
            [70, 80, 1],
            # ... (17 keypoints)
        ]
    ]),
    "keypoint_weights": torch.ones(batch_size, num_keypoints, 1),
    "class_labels": torch.tensor([
        [0, 1],
        [1, 2],
        [2, 0]
    ]),
    "bboxes": torch.tensor([
        [
            [10, 20, 50, 60],
            [30, 40, 70, 80]
        ],
        [
            [50, 60, 90, 100],
            [20, 30, 60, 70]
        ],
        [
            [40, 50, 80, 90],
            [60, 70, 100, 110]
        ]
    ]),
    "image_ids": torch.tensor([0, 1, 2])
}

# 输出形状
print("batch['image'] shape:", batch["image"].shape)
print("batch['keypoints'] shape:", batch["keypoints"].shape)
print("batch['keypoint_weights'] shape:", batch["keypoint_weights"].shape)
print("batch['class_labels'] shape:", batch["class_labels"].shape)
print("batch['bboxes'] shape:", batch["bboxes"].shape)
print("batch['image_ids'] shape:", batch["image_ids"].shape)

# 输出内容
print("batch['image']:\n", batch["image"])
print("batch['keypoints']:\n", batch["keypoints"])
print("batch['keypoint_weights']:\n", batch["keypoint_weights"])
print("batch['class_labels']:\n", batch["class_labels"])
print("batch['bboxes']:\n", batch["bboxes"])
print("batch['image_ids']:\n", batch["image_ids"])
