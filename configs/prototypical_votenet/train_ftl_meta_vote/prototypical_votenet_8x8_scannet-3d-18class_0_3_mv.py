_base_ = [
    '../../_base_/datasets/fs_scannet/split0_fs/scannet-3d-18class_3_shot.py', '../../_base_/models/prototypical_votenet.py',
    '../../_base_/schedules/schedule_3x-fs.py', '../../_base_/default_runtime.py'
]
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')


# model settings
model = dict(
    bbox_head=dict(
        num_classes=18,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=18,
            num_dir_bins=1,
            with_rot=False,
            mean_sizes = [[0.76966727, 0.8116021, 0.92573744],
                          [1.876858, 1.8425595, 1.1931566],
                          [0.61328, 0.6148609, 0.7182701],
                          [1.62373384, 1.59570182, 0.9543768],
                          [0.97949594, 1.0675149, 0.6329687],
                          [0.531663, 0.5955577, 1.7500148],
                          [0.97286712, 1.9191374, 1.32407892],
                          [1.43204941, 1.19174419, 1.06299813],
                          [0.21132214, 0.4206159, 0.5372846],
                          [1.4440073, 1.8970833, 0.26985747],
                          [1.0294262, 1.4040797, 0.87554324],
                          [1.3766412, 0.65521795, 1.6813129],
                          [0.6650819, 0.71111923, 1.298853],
                          [0.41999173, 0.37906948, 1.7513971],
                          [0.57210855, 0.65708703, 0.78886991],
                          [0.50867593, 0.50656086, 0.30136237],
                          [1.02678736, 1.07852574, 0.52460438],
                          [0.37459035, 0.52920857, 0.51707938]])))

# yapf:disable
log_config = dict(interval=30)

