import json
import os
import pickle
from glob import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

#
# class ExemplarDataset(Dataset):
#
#     def __init__(self, data):
#         labels = []
#         for y, P_y in enumerate(data):
#             label = [y] * len(P_y)
#             labels.extend(label)
#         self.data = np.concatenate(data, axis=0)
#         self.labels = np.array(labels)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#
#         sample = self.data[idx]
#         label = self.labels[idx]
#         #sample = transforms.ToPILImage(sample)
#
#         return sample, label
cat = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle',
       'bowl', 'bus', 'cabinet', 'can', 'camera', 'cap',  'car', 'cellphone', 'chair', 'clock',
       'dishwasher', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'keyboard', 'knife',
        'lamp', 'laptop', 'mailbox', 'microphone', 'microwave', 'monitor', 'motorcycle',
        'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote_control',
        'rifle', 'rocket', 'skateboard', 'sofa', 'speaker', 'stove','table',
        'telephone', 'tin_can', 'tower', 'train', 'vessel', 'washer']


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
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


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class ShapeNetDataLoader(Dataset):
    def __init__(self, root, args, class_cls, dataset_name='modelnet40', class_choices=cat,
                 num_points=2048, split='train', load_name=True, load_file=True,
                 segmentation=False, random_rotate=False, random_jitter=False,
                 random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart',
                                        'modelnet10', 'modelnet40', 'shapenetpartpart']
        assert num_points <= 2048

        if dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        if dataset_name not in ['shapenetpart'] and segmentation == True:
            raise AssertionError

        self.root = os.path.join(root, dataset_name + '_' + '*hdf5_2048')
        # self.root = '/home/cuiheng/chdir/VNPCCL2.0/data/shapenetcorev2_hdf5_2048'
        self.class_cls = class_cls
        self.dataset_name = dataset_name
        self.class_choices = class_choices
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.segmentation = segmentation
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        self.uniform = args.use_uniform_sample
        self.npoints = 1024
        self.process_data = args.process_data
        self.new_classes = args.new_classes
        self.start_classes = args.start_classes

        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train', 'trainval', 'all']:
            self.get_path('train')
        if self.dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            if self.split in ['val', 'trainval', 'all']:
                self.get_path('val')
        if self.split in ['test', 'all']:
            self.get_path('test')

        self.path_h5py_all.sort()
        # if self.split == 'train':
        #     self.path_h5py_all = [
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train0.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train10.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train11.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train12.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train13.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train14.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train15.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train16.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train17.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train1.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train2.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train3.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train4.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train5.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train6.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train7.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train8.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/train9.h5']
        # else:
        #     self.path_h5py_all = [
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/test0.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/test1.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/test2.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/test3.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/test4.h5',
        #         '/home/huangyanhui/wxh/VNPCCL2.0/data/shapenetcorev2_hdf5_2048/test5.h5']
        data, label, seg = self.load_h5py(self.path_h5py_all)

        if self.load_name or self.class_choices is not None:
            self.path_name_all.sort()
            self.name = np.array(self.load_json(self.path_name_all))  # load label name

        if self.load_file:
            self.path_file_all.sort()
            self.file = np.array(self.load_json(self.path_file_all))  # load file name

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)
        if self.segmentation:
            self.seg = np.concatenate(seg, axis=0)

        self.cat = [cat[i] for i in self.class_cls]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.datas = []
        self.labels = []
        self.names = []
        # if self.class_choices != None:
        # for class_choice in class_choices:
        # indices = (self.name == class_choice)
        # self.data1 = self.data[indices]
        # self.label1 = self.label[indices]
        # self.name1 = self.name[indices]
        # if self.segmentation:
        #     self.seg = self.seg[indices]
        #     id_choice = shapenetpart_cat2id[class_choice]
        #     self.seg_num_all = shapenetpart_seg_num[id_choice]
        #     self.seg_start_index = shapenetpart_seg_start_index[id_choice]
        # if self.load_file:
        #     self.file = self.file[indices]

        #     self.datas.append(self.data[self.name == class_choice])
        #     self.labels.append(self.label[self.name == class_choice])
        #     self.names.append(self.name[self.name == class_choice])
        #
        # self.datas = np.concatenate(self.datas, axis=0)
        # self.labels = np.concatenate(self.labels, axis=0)
        # self.names = np.concatenate(self.names, axis=0)
        #
        # self.points = []
        # self.name1 = []
        # for name in cat:
        #     if name in self.cat:
        #         indices = (self.names == name)
        #         self.points.append(self.datas[indices])
        #         self.name1.append(self.names[indices])
        # self.points = np.concatenate(self.points, axis=0)
        # self.name1 = np.concatenate(self.name1, axis=0)
        #
        # for n in self.name1:
        #     lb = self.classes[n]
        #     pc = self.points[:sum(self.name1 == n)]
        # point_set = farthest_point_sample(pc, 1024)

        if self.uniform:
            self.save_path = os.path.join(root, 'shapenetcorev2_%d_%s_%s_%dpts_fps.dat' % (
                len(self.classes), list(self.classes.keys())[list(self.classes.values()).index(0)], split,
                self.npoints))
        else:
            self.save_path = os.path.join(root, 'shapenetcorev2_%d_%s_%s_%dpts_normal.dat' % (
            len(self.classes), list(self.classes.keys())[list(self.classes.values()).index(0)], split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.point_sets = []
                self.labels = []
                for index in tqdm(range(len(self.label)), total=len(self.label)):
                    if (self.name[index] in self.cat):
                        label = self.classes[self.name[index]]
                        point_set = self.data[index]

                        point_set = farthest_point_sample(point_set, 1024)
                        point_set = pc_normalize(point_set)

                        self.point_sets.append(point_set)
                        self.labels.append(label)

                self.train_data = np.array(self.point_sets)
                self.train_label = np.array(self.labels)

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


    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5' % type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json' % type)
            self.path_name_all += glob(path_json)
        if self.load_file:
            path_json = os.path.join(self.root, '%s*_id2file.json' % type)
            self.path_file_all += glob(path_json)
        return

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            if self.segmentation:
                seg = f['seg'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            if self.segmentation:
                all_seg.append(seg)
        return all_data, all_label, all_seg

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j = open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, index):
        if self.split == 'train':
            pc, label = self.train_data[index], self.train_label[index]

        else:
            pc, label = self.train_data[index], self.train_label[index]

        return pc, label

    def __len__(self):
        return len(self.train_data)

    def get_pc_class(self, label, t):
        if t == 0:
            return self.train_data[np.array(self.train_label) == label]
        else:
            label = label - (self.start_classes + self.new_classes * (t - 1))
            return self.train_data[np.array(self.train_label) == label]
