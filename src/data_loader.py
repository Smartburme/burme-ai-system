import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config.settings import Config

class MyanmarTextDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length):
        self.config = Config()
        self.data = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = int(self.data.iloc[idx]['label'])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
