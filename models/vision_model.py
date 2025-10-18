import torch
import torch.nn as nn
import torchvision.models as models

class VisionClassifier(nn.Module):
    """ResNet50-based image classifier for plant disease detection"""
    
    def __init__(self, num_classes=15, pretrained=True):
        super(VisionClassifier, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        self.num_classes = num_classes
    
    def forward(self, x):
        logits = self.backbone(x)
        return logits
    
    def get_embedding(self, x):
        """Extract embedding before classification head"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        embedding = x.view(x.size(0), -1)
        return embedding

def create_vision_model(num_classes=15):
    """Factory function to create vision model"""
    model = VisionClassifier(num_classes=num_classes)
    return model
