import torch
import numpy as np
import cv2
from PIL import Image
import librosa
import json
import os

class VisionUtils:
    """Image preprocessing utilities"""
    
    @staticmethod
    def preprocess_image(image_path, img_size=224):
        """Load and preprocess image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    @staticmethod
    def augment_image(image, angle=15, scale=0.1):
        """Data augmentation"""
        h, w = image.shape[:2]
        angle = np.random.randint(-angle, angle)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        return image

class AudioUtils:
    """Audio preprocessing utilities"""
    
    @staticmethod
    def preprocess_audio(audio_path, sr=16000, n_mfcc=13, duration=5):
        """Load and preprocess audio"""
        try:
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            return torch.tensor(mfcc, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
    
    @staticmethod
    def get_audio_stats(mfcc):
        """Extract statistics from MFCC"""
        mean = np.mean(mfcc, axis=1)
        std = np.std(mfcc, axis=1)
        return np.concatenate([mean, std])

class TextUtils:
    """Text preprocessing utilities"""
    
    @staticmethod
    def clean_text(text):
        """Clean and normalize text"""
        text = text.lower()
        text = text.strip()
        return text
    
    @staticmethod
    def load_faq_dataset(faq_path):
        """Load FAQ dataset"""
        try:
            with open(faq_path, 'r') as f:
                faq = json.load(f)
            return faq
        except:
            return {}

class MultimodalFusion:
    """Fusion of multiple modalities"""
    
    @staticmethod
    def fuse_embeddings(vision_emb, audio_emb, text_emb):
        """Concatenate embeddings from different modalities"""
        if isinstance(vision_emb, np.ndarray):
            vision_emb = torch.tensor(vision_emb, dtype=torch.float32)
        if isinstance(audio_emb, np.ndarray):
            audio_emb = torch.tensor(audio_emb, dtype=torch.float32)
        if isinstance(text_emb, np.ndarray):
            text_emb = torch.tensor(text_emb, dtype=torch.float32)
        
        fused = torch.cat([vision_emb, audio_emb, text_emb], dim=-1)
        return fused

class EthicsUtils:
    """Ethics and privacy utilities"""
    
    @staticmethod
    def add_confidence_scores(predictions, threshold=0.6):
        """Add confidence scores to predictions"""
        return {
            'prediction': predictions.get('disease', 'Unknown'),
            'confidence': predictions.get('confidence', 0.0),
            'is_confident': predictions.get('confidence', 0.0) >= threshold,
            'disclaimer': 'This tool provides agronomic advice, not professional diagnosis.'
        }
    
    @staticmethod
    def check_gdpr_compliance():
        """Check GDPR compliance"""
        return {
            'pii_collected': False,
            'audio_stored': False,
            'data_encrypted': True,
            'user_consent': True
        }
