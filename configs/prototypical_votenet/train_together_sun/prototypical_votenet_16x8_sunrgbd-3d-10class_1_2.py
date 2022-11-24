_base_ = [
    '../../_base_/datasets/fs_sunrgbd/split1/sunrgbd-3d-10class_2_shot.py', '../../_base_/models/prototypical_votenet.py',
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
                [2.31319,  1.786194, 0.963896],
                [0.595446, 1.121217, 0.563636],
                [0.923508, 1.867419, 0.845495],
                [0.591958, 0.552978, 0.827272],
                [0.691651, 0.370706, 0.770454],
                [0.69519, 1.346299, 0.736364],
                [0.528526, 1.002642, 1.172878],
                [0.508871, 0.693651, 0.709166],
                [0.404671, 1.071108, 1.688889],
                [0.76584, 1.398258, 0.472728]
            ]),
    ))
