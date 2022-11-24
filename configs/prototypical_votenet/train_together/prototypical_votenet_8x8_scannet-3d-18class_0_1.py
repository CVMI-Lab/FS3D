_base_ = [
    '../../_base_/datasets/fs_scannet/split0/scannet-3d-18class_1_shot.py', '../../_base_/models/prototypical_votenet.py',
    '../../_base_/schedules/schedule_3x.py', '../../_base_/default_runtime.py'
]
#few_shot_class = ('bathtub', 'toilet', 'bookshelf', 'sofa', 'window', 'garbagebin')
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
                        [1.876858, 1.8425595, 1.1931566],
                        [0.61328, 0.6148609, 0.7182701],
                        [3.48712861, 4.02809171, 1.09827888],
                        [0.97949594, 1.0675149, 0.6329687],
                        [0.531663, 0.5955577, 1.7500148],
                        [0.24795047, 1.02802394, 0.89268166],
                        [1.36156682, 1.38509727, 2.01948537],
                        [0.21132214, 0.4206159, 0.5372846],
                        [1.4440073, 1.8970833, 0.26985747],
                        [1.0294262, 1.4040797, 0.87554324],
                        [1.3766412, 0.65521795, 1.6813129],
                        [0.6650819, 0.71111923, 1.298853],
                        [0.41999173, 0.37906948, 1.7513971],
                        [0.68297335, 0.53146613, 0.74332048],
                        [0.50867593, 0.50656086, 0.30136237],
                        [1.28962977, 0.75683334, 0.49129029],
                        [0.5203573,  0.35548513, 0.86446355]])))


# yapf:disable
log_config = dict(interval=30)
# yapf:enable
load_from = None

