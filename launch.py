import torch
from transformers import pipeline


def loadmodel(logger):
    """Get model from cloud object storage."""
   
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")
    model = pipeline(model="openai/whisper-medium", device=device)
    return model  

def preprocessing(features, logger):
    """ Applies preprocessing techniques to the raw data"""
    logger.info("no preprocessing required")
    return False
    
def predict(features, model, logger):
    """Predicts the results for the given inputs"""
    logger.info("model prediction")
    prediction = model(features)["text"]
    return prediction
