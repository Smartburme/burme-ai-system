import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate(self, dataloader):
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k:v.to(self.device) for k,v in batch.items() 
                         if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        return classification_report(actual_labels, predictions, output_dict=True)
