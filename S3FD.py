import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Norm(nn.Module):
    def __init__(self, n_channels):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.weight = nn.Parameter(torch.ones(n_channels))

    def forward(self, x):
        # 确保 self.weight 的形状与 x 的通道数一致
        weight = self.weight.view(1, self.n_channels, 1, 1)
        x = x / (torch.sqrt(torch.sum(x**2, dim=1, keepdim=True)) + 1e-10) * weight
        return x

class S3FD(nn.Module):
    def __init__(self, device="cuda"):
        super(S3FD, self).__init__()
        self.minus = torch.tensor([104, 117, 123], dtype=torch.float32)
        self.device = device

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256)
        self.conv4_3_norm = L2Norm(512)
        self.conv5_3_norm = L2Norm(512)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)

        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x - self.minus.view(1, 3, 1, 1).to(x.device)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        f3_3 = x
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        f4_3 = x
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        f5_3 = x
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        ffc7 = x

        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        f6_2 = x

        x = F.relu(self.conv7_1(x))
        x = F.relu(self.conv7_2(x))
        f7_2 = x

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)

        cls2 = F.softmax(self.conv4_3_norm_mbox_conf(f4_3), dim=1)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)

        cls3 = F.softmax(self.conv5_3_norm_mbox_conf(f5_3), dim=1)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)

        cls4 = F.softmax(self.fc7_mbox_conf(ffc7), dim=1)
        reg4 = self.fc7_mbox_loc(ffc7)

        cls5 = F.softmax(self.conv6_2_mbox_conf(f6_2), dim=1)
        reg5 = self.conv6_2_mbox_loc(f6_2)

        cls6 = F.softmax(self.conv7_2_mbox_conf(f7_2), dim=1)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # max-out background label logic
        # 假设 cls1 的形状为 [batch_size, height, width, num_classes]
        # 这里需要根据实际的张量维度进行调整
        # 以下代码逻辑是基于 TensorFlow 实现中的维度顺序进行的模拟

        # 对 cls1 的前三个通道取最大值
        # PyTorch 中取通道维度上的最大值略有不同
        # 需要确保 dim 参数正确指向通道维度
        # 假设输入是 (batch_size, num_channels, height, width)
        # 那么 dim=1 对应通道维度
        bmax = torch.max(torch.max(cls1[:, 0:1, :, :], cls1[:, 1:2, :, :]), cls1[:, 2:3, :, :])

        # 将最大值与第四个通道拼接
        # 需要确保拼接的维度正确
        # 这里假设是沿着通道维度进行拼接
        cls1 = torch.cat([bmax, cls1[:, 3:4, :, :]], dim=1)

        # 对拼接后的张量进行 softmax 激活
        cls1 = F.softmax(cls1, dim=1)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]
