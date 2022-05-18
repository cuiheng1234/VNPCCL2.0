import numpy as np
import torch
from torch.autograd import Variable

exemplar_sets = []


def icarl_construct_exemplar_set(model, pc_set, m):
    model.eval()
    # Compute and cache features for each example
    features = []
    with torch.no_grad():
        for pc in pc_set:
            x = Variable(torch.from_numpy(pc)).cuda()
            # print('点云的变量是什么类型的呢？？ ')
            x = x.unsqueeze(0)
            # x = x[:, :, :3]
            x = x.transpose(2, 1)
            feat = model.forward(x, rd=True).data.cpu().numpy()
            feat = feat / np.linalg.norm(feat)  # Normalize
            features.append(feat[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

        exemplar_set = []
        exemplar_features = []  # list of Variables of shape (feature_size,)
        exemplar_dist = []
        for k in range(int(m)):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))

            idx = np.random.randint(0, features.shape[0])

            exemplar_dist.append(dist[idx])
            exemplar_set.append(pc_set[idx])
            exemplar_features.append(features[idx])
            features[idx, :] = 0.0

        # random exemplar selection
        exemplar_dist = np.array(exemplar_dist)
        exemplar_set = np.array(exemplar_set)
        ind = exemplar_dist.argsort()
        exemplar_set = exemplar_set[ind]

        exemplar_sets.append(np.array(exemplar_set))
    print('exemplar set shape: ', len(exemplar_set))
    return exemplar_sets


def icarl_construct_exemplar_sets(m, exemplar_sets):
    for y, P_y in enumerate(exemplar_sets):
        exemplar_sets[y] = P_y[:m]
    return exemplar_sets
