_base_ = [
    '../../_base_/datasets/fs_scannet/split1/scannet-3d-18class_5_shot.py', '../../_base_/models/prototypical_votenet.py',
    '../../_base_/schedules/schedule_3x.py', '../../_base_/default_runtime.py'
]

# few_shot_class = ('showercurtrain', 'counter', 'bed', 'desk', 'table', 'door')

# model settings
model = dict(
    bbox_head=dict(
        num_classes=18,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=18,
            num_dir_bins=1,
            with_rot=False,
            mean_sizes=[[0.76966727, 0.8116021, 0.92573744],
                        [1.83346831, 1.96696474, 1.18555685],
                        [0.61328, 0.6148609, 0.7182701],
                        [1.3955007, 1.5121545, 0.83443564],
                        [0.84235485, 1.24126494, 0.70496498],
                        [0.48046361, 0.58226301, 1.56031816],
                        [0.9624706, 0.72462326, 1.1481868],
                        [0.83221924, 1.0490936, 1.6875663],
                        [0.21132214, 0.4206159, 0.5372846],
                        [0.72107936, 2.16533782, 0.26353514],
                        [0.98859121, 1.48774168, 0.79578456],
                        [1.3766412, 0.65521795, 1.6813129],
                        [0.6650819, 0.71111923, 1.298853],
                        [0.23482446, 0.23133101, 1.64376909],
                        [0.59359556, 0.5912492, 0.73919016],
                        [0.50867593, 0.50656086, 0.30136237],
                        [1.1511526, 1.0546296, 0.49706793],
                        [0.47535285, 0.49249494, 0.5802117]])))

# yapf:disable
log_config = dict(interval=30)
# yapf:enable
