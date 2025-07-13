import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from model import MyanmarXLMRoberta
from preprocessing import MyanmarTextPreprocessor
from config.settings import Config

class Trainer:
    def __init__(self):
        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MyanmarXLMRoberta(self.config).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.preprocessor = MyanmarTextPreprocessor()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            self.optimizer.zero_grad()
            inputs = {k:v.to(self.device) for k,v in batch.items()}
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
