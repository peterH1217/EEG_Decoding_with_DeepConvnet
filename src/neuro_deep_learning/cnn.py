import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, n_channels, n_classes, input_window_samples):
        super().__init__()
        self.elu = nn.ELU()

        # ===== Block 1: Temporal + Spatial =====
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=25,
            kernel_size=(1, 10),
            bias=False
        )

        self.spatial_conv = nn.Conv2d(
            in_channels=25,
            out_channels=25,
            kernel_size=(n_channels, 1),
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(p=0.5)

        # ===== Block 2 =====
        self.conv2 = nn.Conv2d(25, 50, kernel_size=(1, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop2 = nn.Dropout(p=0.5)

        # ===== Block 3 =====
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(1, 10), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop3 = nn.Dropout(p=0.5)

        # ===== Block 4 =====
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(1, 10), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.drop4 = nn.Dropout(p=0.5)

        # ===== Classifier =====
        self.n_outputs = self._get_flattened_size(n_channels, input_window_samples)
        self.classifier = nn.Linear(self.n_outputs, n_classes)

    def _get_flattened_size(self, n_channels, input_window_samples):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, input_window_samples)

            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.bn1(x)
            x = self.elu(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.elu(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.elu(x)
            x = self.pool3(x)

            x = self.conv4(x)
            x = self.bn4(x)
            x = self.elu(x)
            x = self.pool4(x)

            return x.numel()

    def forward(self, x):
        # Input: (Batch, 1, Channels, Time)

        # ----- Block 1 -----
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # ----- Block 2 -----
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # ----- Block 3 -----
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # ----- Block 4 -----
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.pool4(x)
        x = self.drop4(x)

        # ----- Classifier -----
        x = x.view(x.size(0), -1)
        return self.classifier(x)