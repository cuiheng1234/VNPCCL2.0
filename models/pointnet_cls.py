import torch.nn as nn
import torch.nn.functional as F

from models.modified_linear import CosineLinear, SplitCosineLinear
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, args,num_classes=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = CosineLinear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def get_output_dim(self):
        return self.fc3.out_features

    def change_output_dim(self, new_dim, second_iter=False):

        if second_iter:
            in_features = self.fc3.in_features
            out_features1 = self.fc3.fc1.out_features
            out_features2 = self.fc3.fc2.out_features
            print("in_features:", in_features, "out_features1:", \
                  out_features1, "out_features2:", out_features2)
            new_fc = SplitCosineLinear(in_features, out_features1 + out_features2, out_features2)
            new_fc.fc1.weight.data[:out_features1] = self.fc3.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = self.fc3.fc2.weight.data
            new_fc.sigma.data = self.fc3.sigma.data
            self.fc3 = new_fc
            new_out_features = new_dim
            self.n_classes = new_out_features

        else:
            in_features = self.fc3.in_features
            out_features = self.fc3.out_features

            print("in_features:", in_features, "out_features:", out_features)
            new_out_features = new_dim
            num_new_classes = new_dim - out_features
            new_fc = SplitCosineLinear(in_features, out_features, num_new_classes)

            new_fc.fc1.weight.data = self.fc3.weight.data
            new_fc.sigma.data = self.fc3.sigma.data
            self.fc3 = new_fc
            self.n_classes = new_out_features

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, feat=False, rd=False):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))

        if rd:
            return F.normalize(x, p=2, dim=1)

        if feat:
            pass
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x
