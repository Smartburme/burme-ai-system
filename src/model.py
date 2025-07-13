import torch.nn as nn
from transformers import XLMRobertaModel

class MyanmarXLMRoberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.xlmr = XLMRobertaModel.from_pretrained(config.MODEL_NAME)
        self.classifier = nn.Linear(self.xlmr.config.hidden_size, config.NUM_LABELS)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.xlmr(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
