import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm import create_model

#Baseline model
class ConvNet(nn.Module):
    """
    Convolutional Neural Network model for knife classification.
    
    Args:
        num_classes (int): Number of classes for classification.
    """
    
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the ConvNet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Fine-tuned ResNet model, replace the last layer with a new fully connected layer
class FineTunedResNet(nn.Module):
    """
    Fine-tuned ResNet model for knife classification.
    
    Args:
        num_classes (int): Number of output classes.
    """
      
    def __init__(self, num_classes):
        super(FineTunedResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.resnet(x)
        return x

#Fine-tuned ResNet model, replace the last layer with a new fully connected layer, 
# with an additional fully connected layer
class FineTunedResNetV1(nn.Module):
    """
    Fine-tuned ResNetV1 model for knife classification.
    
    Args:
        num_classes (int): The number of classes for classification.
    """
    
    def __init__(self, num_classes):
        super(FineTunedResNetV1, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1024)
        self.fc1 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        return x

class FineTunedDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedDenseNet, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for param in self.densenet.parameters():
            param.requires_grad = False
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        
    def forward(self, x):
        x = self.densenet(x)
        return x

# class FineTunedEffNetV2(nn.Module):
#     def __init__(self, num_classes):
#         super(FineTunedEffNetV2, self).__init__()
#         self.effnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
#         for param in self.effnet.parameters():
#             param.requires_grad = False
#         self.effnet.classifier = nn.Linear(self.effnet[-1].in_features, 1024)
#         self.fc1 = nn.Linear(1024, num_classes)
        
#     def forward(self, x):   
#         x = self.effnet(x)
#         x = self.fc1(x)
#         return x



class FineTunedEffNetV2(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedEffNetV2, self).__init__()
        self.effnet = create_model('tf_efficientnetv2_m', pretrained=True, num_classes=num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        
        # # Freeze all layers
        # for param in self.effnet.parameters():
        #     param.requires_grad = False

        # # Unfreeze last 2 layers
        # for param in self.effnet.blocks[-1].parameters():
        #     param.requires_grad = True

        # for param in self.effnet.classifier.parameters():
        #     param.requires_grad = True
        
    def forward(self, x):   
        x = self.effnet(x)
        x = self.bn(x)
        return x