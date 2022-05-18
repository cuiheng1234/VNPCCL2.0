import h5py
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


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


class ScanObjectNNDataLoader(Dataset):
    def __init__(self, root,  args, class_cls, split='train'):
        self.root = root
        self.class_cls = class_cls
        self.split = split
        self.start_classes = args.start_classes
        self.new_classes = args.new_classes

        # self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.classes = dict(zip(self.class_cls, range(len(self.class_cls))))

        if self.split == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        train_data = []
        train_label = []
        for i in range(0,15):
            print('类别：{0}, 数量：{1}'.format(i, sum(self.labels==i)))
        for cls in self.class_cls:


            indx = cls == self.labels

            self.train_data1 = self.points[indx]
            list = range(0, 11416)
            for i in tqdm(range(0, len(self.train_data1)), total=len(self.train_data1)):
                pc = self.train_data1[i]
                points = farthest_point_sample(pc, 1024)
                train_data.append(points)
            train_label.append(self.labels[indx])
        train_data1 = np.array(train_data)
        train_label1 = np.concatenate(np.array(train_label))

        self.train_data = train_data1
        # [self.classes.get(key) for key in train_label1]
        self.train_label = [self.classes.get(key) for key in train_label1]





    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        if self.split == 'train':
            pc, label = self.train_data[index], self.train_label[index]

        else:
            pc, label = self.train_data[index], self.train_label[index]

        return pc, label

        print('complete')

    def get_pc_class(self, label, t):
        if t == 0:
            return self.train_data[np.array(self.train_label) == label]
        else:
            label = label - (self.start_classes + self.new_classes * (t - 1))
            return self.train_data[np.array(self.train_label) == label]
