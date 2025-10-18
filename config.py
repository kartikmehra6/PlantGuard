# Configuration for PlantGuard

DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew',
    'Cherry___healthy', 'Corn___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_Black_Measles',
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot', 'Grape___healthy'
]

NUM_CLASSES = len(DISEASE_CLASSES)
IMG_SIZE = 224
BATCH_SIZE = 32
SAMPLE_RATE = 16000
N_MFCC = 13
CONFIDENCE_THRESHOLD = 0.6
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128

VISION_MODEL_PATH = 'models/vision_model.pth'
AUDIO_MODEL_PATH = 'models/audio_model.pth'
TEXT_MODEL_PATH = 'models/text_model'

STORE_USER_DATA = False
STORE_AUDIO = False
