import torch
import torch.nn as nn

class MultimodalFusionModel(nn.Module):
    """Multi-modal fusion architecture"""
    
    def __init__(self, vision_emb_size=2048, audio_emb_size=256, 
                 text_emb_size=768, num_classes=15):
        super(MultimodalFusionModel, self).__init__()
        
        total_emb_size = vision_emb_size + audio_emb_size + text_emb_size
        
        self.fusion_network = nn.Sequential(
            nn.Linear(total_emb_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, vision_emb, audio_emb, text_emb, weights=None):
        """Forward pass with optional modality weighting"""
        if weights is None:
            weights = {'vision': 1.0, 'audio': 1.0, 'text': 1.0}
        
        vision_emb = vision_emb * weights['vision']
        audio_emb = audio_emb * weights['audio']
        text_emb = text_emb * weights['text']
        
        fused = torch.cat([vision_emb, audio_emb, text_emb], dim=-1)
        logits = self.fusion_network(fused)
        return logits

def create_multimodal_model(vision_emb_size=2048, audio_emb_size=256,
                           text_emb_size=768, num_classes=15):
    """Factory function to create multimodal model"""
    model = MultimodalFusionModel(vision_emb_size=vision_emb_size,
                                  audio_emb_size=audio_emb_size,
                                  text_emb_size=text_emb_size,
                                  num_classes=num_classes)
    return model
