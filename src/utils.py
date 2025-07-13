import os
import json
import torch
from datetime import datetime

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def save_results(results, experiment_name):
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/{experiment_name}_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
