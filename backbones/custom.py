import torch.nn as nn


class CustomBackbone(nn.Module):
    def __init__(self, num_classes, use_stride=False):
        super(CustomBackbone, self).__init__()
        self.use_stride = use_stride

        # Immagini di input: 3x32x32 (CIFAR-10/CIFAR-100)

        # Block 1: Primo strato convoluzionale
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # -> 64x32x32

        # Block 2: Secondo strato convoluzionale
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ) # -> 128x32x32

        # Block 3: Terzo strato convoluzionale (con downsampling)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2 if use_stride else 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if not use_stride else nn.Identity()
        ) # -> 256x16x16

        # Block 4: Quarto strato convoluzionale
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) # -> 256x16x16

        # Block 5: Quinto strato convoluzionale (con downsampling)
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2 if use_stride else 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if not use_stride else nn.Identity()
        ) # -> 512x8x8

        # Block 6: Sesto strato convoluzionale
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ) # -> 512x8x8

        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.classifier(x)
        return x
