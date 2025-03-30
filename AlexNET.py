import torch
import torch.nn as nn

class AlexNET(nn.Module):
    def __init__(self, num_classes):
        super(AlexNET, self).__init__()

        self.features = nn.Sequential(
            # Layer 1: Conv -> ReLU -> LRN -> MaxPool
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),  # Output: 55x55x96
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # Output: 27x27x96

            # Layer 2: Conv -> ReLU -> LRN -> MaxPool
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),  # Output: 27x27x256
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # Output: 13x13x256

            # Layer 3: Conv -> ReLU
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # Output: 13x13x384
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            # Layer 4: Conv -> ReLU
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # Output: 13x13x384
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            # Layer 5: Conv -> ReLU -> MaxPool
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # Output: 13x13x256
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # Output: 6x6x256
        )

        self.classifier = nn.Sequential(
            # Layer 6: Dropout -> FC -> ReLU
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),

            # Layer 7: Dropout -> FC -> ReLU
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),

            # Layer 8: FC (output layer)
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
        x = self.classifier(x)
        return x

def get_AlexNET_model(num_classes):
    model = AlexNET(num_classes)
    return model