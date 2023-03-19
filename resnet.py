import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class ResNetModel(ResNet):
    def __init__(self, num_classes=6):
        super(ResNetModel, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define the device
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
device = torch.device('cpu')
# Define the path to the saved model
model_path = './resnet-model/pytorch_resnet.pt'
# Load the saved model
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model = ResNetModel(num_classes=6)
model.load_state_dict(checkpoint['model_state_dict'])

# # Move the model to the device
# model.to(device)

# Define label names
label_names = ['regular', 'help', 'robbery', 'sexual', 'theft', 'violence']