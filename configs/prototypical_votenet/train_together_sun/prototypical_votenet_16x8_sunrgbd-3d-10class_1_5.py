_base_ = [
    '../../_base_/datasets/fs_sunrgbd/split1/sunrgbd-3d-10class_5_shot.py', '../../_base_/models/prototypical_votenet.py',
    '../../_base_/schedules/schedule_3x_sun.py', '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    bbox_head=dict(
        num_classes=10,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=10,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [2.2834456, 1.6283352, 1.0298576],
                [1.0146096, 1.3535872, 0.6813292],
                [0.923508, 1.867419, 0.845495],
                [0.591958, 0.552978, 0.827272],
                [0.7441548, 0.4518376, 0.77996],
                [0.69519, 1.346299, 0.736364],
                [0.528526, 1.002642, 1.172878],
                [0.5370856, 0.6146416, 0.6581872],
                [0.404671, 1.071108, 1.688889],
                [0.76584, 1.398258, 0.472728]
            ]),
    ))
