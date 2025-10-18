import torch
import torch.nn as nn

class AudioClassifier(nn.Module):
    """CNN-LSTM classifier for audio symptom descriptions"""
    
    def __init__(self, n_mfcc=13, num_classes=15, hidden_size=128):
        super(AudioClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        
        logits = self.fc(x)
        return logits
    
    def get_embedding(self, x):
        """Extract embedding before classification head"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        embedding = x[:, -1, :]
        return embedding

def create_audio_model(n_mfcc=13, num_classes=15, hidden_size=128):
    """Factory function to create audio model"""
    model = AudioClassifier(n_mfcc=n_mfcc, num_classes=num_classes, hidden_size=hidden_size)
    return model
