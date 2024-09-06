JOINTS_LIST = ['Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle',
               'Right Wrist', 'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow',
               'Left Wrist', 'Head-bottom','Nose','Head-top']
DATASET_PATH_TRAIN = r'E:\pc\目标识别相关1\PoseTrack2017\posetrack_data\annotations\train'
DATASET_PATH_VAL = r'E:\pc\目标识别相关1\PoseTrack2017\posetrack_data\annotations\val'
TEMPORAL = 5  # The length of frames sequence
EPOCHS = 5
LR = 1e-4
TRAIN_BS = 8
EVAL_BS = 2
WEIGHT_DECAY = 0.01
SCH_GAMMA = 0.4
SCH_STEP = 1
