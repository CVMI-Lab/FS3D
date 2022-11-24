import torch

def feature_norm(feature):
    features_norm = torch.norm(feature, p=2, dim=1)
    feature = feature.div(features_norm.unsqueeze(1))
    return feature

def cls_gather(x, pts_semantic_masks, pts_instance_masks, points_xyzs, seed_indices, class_names=None, background_flag=17):

    points = x['fp_xyz'][-1]
    features = x['fp_features'][-1].detach()
    fp_indices = x['fp_indices'][-1]

    if class_names[0] == 'cabinet':
        num_p = 40000
    else:
        num_p = 20000

    batch_size = features.shape[0]
    batch_list = []
    for ii in range(batch_size):
        feature = features[ii]
        pts_semantic_mask = pts_semantic_masks[ii]
        pts_instance_mask = pts_instance_masks[ii]

        new_dict = {}
        unique_pts_instance_mask = torch.unique(pts_instance_mask)
        binary_pts_instance_mask = pts_instance_mask.new(num_p, ).zero_()
        binary_pts_instance_mask[fp_indices[ii]] = 1

        index = binary_pts_instance_mask.new_ones(fp_indices.shape[-1])
        index = index.cumsum(dim=0) - 1
        binary_index = pts_instance_mask.new(num_p, ).zero_()
        binary_index[fp_indices[ii]] = index

        for i in unique_pts_instance_mask:
            indices = torch.nonzero(pts_instance_mask == i, as_tuple=False).squeeze(-1)
            binary_indices = pts_instance_mask.new(num_p, ).zero_()
            binary_indices[indices] = 1
            binary_indices = torch.nonzero(binary_indices & binary_pts_instance_mask, \
                                           as_tuple=False).squeeze(-1)
            binary_indices = binary_index[binary_indices]
            class_index = pts_semantic_mask[indices[0]]

            if class_index > background_flag:
                continue

            one_instance_feature = feature[:, binary_indices]
            context_features = one_instance_feature

            if context_features.shape[1] > 0:
                context_feature = torch.max(context_features, 1)[0].reshape(1, -1).detach()

            if one_instance_feature.shape[1] < 1:
                continue

            one_instance_feature = torch.max(one_instance_feature, 1)[0].reshape(1, -1)

            class_name = class_names[class_index]

            if class_name not in new_dict.keys():
                new_dict[class_name] = [[], []]
            new_dict[class_name][0].append(one_instance_feature.detach())
            if context_features.shape[1] > 0:
                new_dict[class_name][1].append(context_feature)

        batch_list.append(new_dict)

    return batch_list

def cls_prototype(batch_list, context_compen, num=3, way=6):
    # context_compen = feature_norm(context_compen)
    K_shot = num
    centroids = []
    batch_size = len(batch_list)
    for bs in range(batch_size):
        one_batch = batch_list[bs]
        other_batch = batch_list[0:bs] + batch_list[bs + 1:]
        one_batch_name = one_batch.keys()

        other_dict = {}
        for one_other_batch in other_batch:
            for name, features in one_other_batch.items():
                if name not in other_dict.keys():
                    other_dict[name] = [[], []]
                other_dict[name][0] += features[0]
                other_dict[name][1] += features[1]

        other_dict_center = {}
        for class_name in other_dict.keys():
            class_name_features = other_dict[class_name][0]
            class_name_contexts = other_dict[class_name][1]
            if len(class_name_features) >= K_shot:
                this_k_shot = K_shot
                class_name_features = class_name_features[:this_k_shot]
            else:
                this_k_shot = len(class_name_features)
                class_name_features = class_name_features*K_shot
                class_name_features = class_name_features[:K_shot]

            if len(class_name_contexts) >= K_shot:
                this_k_shot = K_shot
                class_name_contexts = class_name_contexts[:this_k_shot]
            else:
                this_k_shot = len(class_name_contexts)
                compen_num = K_shot - this_k_shot
                class_name_contexts += [context_compen] * compen_num

            instance_features = torch.cat(class_name_features, 0)
            this_center = torch.mean(instance_features, dim=0).reshape(1, -1)

            # this_center = feature_norm(this_center)
            instance_features = feature_norm(instance_features)
            this_center = feature_norm(this_center)

            # this_center = this_center.repeat(K_shot, 1)
            this_context = torch.cat(class_name_contexts, 0)

            if K_shot == 1:
                other_dict_center[class_name] = this_center
            else:
                other_dict_center[class_name] = torch.cat((this_center, instance_features), 0)
            # other_dict_center[class_name] = this_center
            # other_dict_center[class_name] = torch.cat((this_center, instance_features, this_context), 0)

        N_way = way
        one_dict = {}
        one_dict_feature = []
        for name in one_batch_name:
            if name in other_dict_center.keys():
                one_dict[name] = other_dict_center[name]
                one_dict_feature.append(other_dict_center[name])
            else:
                if len(one_batch[name]) > K_shot:
                    feature = one_batch[name][0][:K_shot]
                else:
                    feature = one_batch[name][0]*K_shot
                    feature = feature[:K_shot]

                if len(one_batch[name][1]) >= K_shot:
                    single_context_feature = one_batch[name][1][:K_shot]
                else:
                    this_k_shot = len(one_batch[name][1])
                    single_context_feature = one_batch[name][1]
                    compen_num = K_shot - this_k_shot
                    single_context_feature += [context_compen] * compen_num

                instance_features = torch.cat(feature, 0)
                feature = torch.mean(instance_features, dim=0).reshape(1, -1)
                # feature = feature.repeat(K_shot, 1)
                # feature = feature_norm(feature)
                instance_features = feature_norm(instance_features)
                feature = feature_norm(feature)

                single_context_feature = torch.cat(single_context_feature, 0)

                if K_shot > 1:
                    feature = torch.cat((feature, instance_features), 0)

                # print(feature.shape)
                one_dict[name] = feature
                one_dict_feature.append(feature)
            if len(one_dict) >= N_way:
                break

        for name in other_dict_center.keys():
            if len(one_dict) >= N_way:
                break
            if name not in one_dict.keys():
                one_dict[name] = other_dict_center[name]
                one_dict_feature.append(other_dict_center[name])

        while len(one_dict_feature) < N_way:
            one_dict_feature.append(context_compen)

        one_dict_feature = torch.cat(one_dict_feature, 0).unsqueeze(0)

        centroids.append(one_dict_feature)

    centroids = torch.cat(centroids, 0)

    return centroids

def cls_prototype_support(batch_list, batch_size, num=3, way=6, compen_context=None, few_shot_class=None):

    K_shot = num
    other_batch = batch_list
    one_batch_name = few_shot_class

    other_dict = {}
    for one_other_batch in other_batch:
        for name, features in one_other_batch.items():
            if name not in other_dict.keys():
                other_dict[name] = [[], []]
            other_dict[name][0] += features[0]
            other_dict[name][1] += features[1]

    other_dict_center = {}
    for class_name in other_dict.keys():
        class_name_features = other_dict[class_name][0]
        class_name_contexts = other_dict[class_name][1]
        # print(len(class_name_features))
        if len(class_name_features) >= K_shot:
            this_k_shot = K_shot
            class_name_features = class_name_features[:this_k_shot]
        else:
            this_k_shot = len(class_name_features)
            class_name_features = class_name_features * K_shot
            class_name_features = class_name_features[:K_shot]

        if len(class_name_contexts) >= K_shot:
            this_k_shot = K_shot
            class_name_contexts = class_name_contexts[:this_k_shot]
        else:
            this_k_shot = len(class_name_contexts)
            compen_num = K_shot - this_k_shot
            class_name_contexts += [compen_context] * compen_num

        instance_features = torch.cat(class_name_features, 0)
        this_center = torch.mean(instance_features, dim=0).reshape(1, -1)

        instance_features = feature_norm(instance_features)
        this_center = feature_norm(this_center)

        this_context = torch.cat(class_name_contexts, 0)

        if K_shot == 1:
            other_dict_center[class_name] = this_center
        else:
            other_dict_center[class_name] = torch.cat((this_center, instance_features), 0)
        # other_dict_center[class_name] = this_center
        # other_dict_center[class_name] = torch.cat((this_center, instance_features, this_context), 0)

    N_way = way
    one_dict = {}
    one_dict_feature = []

    for name in one_batch_name:
        if name in other_dict_center.keys():

            one_dict[name] = other_dict_center[name]
            one_dict_feature.append(other_dict_center[name])

    one_dict_feature = torch.cat(one_dict_feature, 0).unsqueeze(0).repeat(batch_size, 1, 1)

    return one_dict_feature