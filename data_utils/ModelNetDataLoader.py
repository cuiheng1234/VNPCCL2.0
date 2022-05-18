import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ExemplarDataset(Dataset):

    def __init__(self, data):
        labels = []
        for y, P_y in enumerate(data):
            label = [y] * len(P_y)
            labels.extend(label)
        self.data = np.concatenate(data, axis=0)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]
        label = self.labels[idx]
        #sample = transforms.ToPILImage(sample)

        return sample, label

class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, class_cls, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.split = split
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.class_cls = class_cls
        self.new_classes = args.new_classes
        self.start_classes = args.start_classes

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        # class_cls = np.array(class_cls)
        self.cat = [[line.rstrip() for line in open(self.catfile)][i] for i in self.class_cls]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]

        index = 0
        self.datapath1 = []
        for name in self.datapath:
            if name[0] in self.cat:
                self.datapath1.append(self.datapath[index])
            index += 1

        print('The size of %s data is %d' % (split, len(self.datapath1)))

        point_sets = []
        labels = []
        if self.use_normals:
            if self.uniform:
                self.save_path = os.path.join(root, 'modelnet%d_%s_%s_%dpts_fps_normal.dat' % (
                len(self.classes), list(self.classes.keys())[list(self.classes.values()).index(0)], split, self.npoints))
            else:
                self.save_path = os.path.join(root, 'modelnet%d_%s_%s_%dpts_normal.dat' % (len(self.classes), list(self.classes.keys())[list(self.classes.values()).index(0)], split,self.npoints))
            if self.process_data:
                if not os.path.exists(self.save_path):
                    print('Processing data %s (only running in the first time)...' % self.save_path)
                    for index in tqdm(range(len(self.datapath1)), total=len(self.datapath1)):
                        fn = self.datapath1[index]
                        label = self.classes[self.datapath1[index][0]]
                        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                        else:
                            point_set = point_set[0:self.npoints, :]

                        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

                        point_sets.append(point_set)
                        labels.append(label)
                    self.train_data = np.array(point_sets)
                    self.train_label = labels

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.train_data, self.train_label], f)
                        f.close()
                else:
                    print('Load processed data from %s...' % self.save_path)
                    with open(self.save_path, 'rb') as f:
                        if args.model == 'vn_pointnet_cls':
                            pc, label = pickle.load(f)
                            self.train_data = pc[:, :, :3]
                            self.train_label = label
                        else:
                            self.train_data, self.train_label = pickle.load(f)

        else:
            if self.uniform:
                self.save_path = os.path.join(root, 'modelnet%d_%s_%s_%dpts_fps_no_normal.dat' % (
                    len(self.classes), list(self.classes.keys())[list(self.classes.values()).index(0)], split,
                    self.npoints))
            else:
                self.save_path = os.path.join(root, 'modelnet%d_%s_%s_%dpts_no_normal.dat' % (
                    len(self.classes), list(self.classes.keys())[list(self.classes.values()).index(0)], split,
                    self.npoints))
            if self.process_data:
                if not os.path.exists(self.save_path):
                    print('Processing data %s (only running in the first time)...' % self.save_path)
                    for index in tqdm(range(len(self.datapath1)), total=len(self.datapath1)):
                        fn = self.datapath1[index]
                        label = self.classes[self.datapath1[index][0]]
                        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                        else:
                            point_set = point_set[0:self.npoints, :]

                        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

                        point_sets.append(point_set[:, :3])
                        labels.append(label)
                    self.train_data = np.array(point_sets)
                    self.train_label = labels

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.train_data, self.train_label], f)
                        f.close()
                else:
                    print('Load processed data from %s...' % self.save_path)
                    with open(self.save_path, 'rb') as f:
                        self.train_data, self.train_label = pickle.load(f)

        print('Data loading completed！！！')

    def __getitem__(self, index):
        if self.split == 'train':
            pc, label = self.train_data[index], self.train_label[index]

        else:
            pc, label = self.train_data[index], self.train_label[index]

        return pc, label

    def __len__(self):
        return len(self.datapath1)

    def get_pc_class(self, label, t):
        if t == 0:
            return self.train_data[np.array(self.train_label) == label]
        else:
            label = label - (self.start_classes + self.new_classes * (t - 1))
            return self.train_data[np.array(self.train_label) == label]
