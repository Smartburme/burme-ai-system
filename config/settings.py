class Config:
    # Text Processing
    MAX_SEQ_LENGTH = 128
    MYANMAR_STOPWORDS = ["များ", "တယ်", "သည်", "၏", "နှင့်", "သော", "၍", "၌"]
    
    # Model Parameters
    MODEL_NAME = "xlm-roberta-base"
    NUM_LABELS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    
    # Paths
    DATA_PATH = "data/processed"
    MODEL_SAVE_PATH = "models/trained_model.bin"
