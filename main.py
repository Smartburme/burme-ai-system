import argparse
from src.train import Trainer
from src.data_loader import MyanmarTextDataset
from src.evaluate import Evaluator
from config.settings import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    args = parser.parse_args()
    
    config = Config()
    trainer = Trainer()
    
    if args.mode == 'train':
        train_dataset = MyanmarTextDataset(
            f"{config.DATA_PATH}/train.csv",
            config.MODEL_NAME,
            config.MAX_SEQ_LENGTH
        )
        trainer.train(train_dataset)
        
    elif args.mode == 'eval':
        eval_dataset = MyanmarTextDataset(
            f"{config.DATA_PATH}/test.csv",
            config.MODEL_NAME,
            config.MAX_SEQ_LENGTH
        )
        evaluator = Evaluator(trainer.model, trainer.device)
        results = evaluator.evaluate(eval_dataset)
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
