import numpy as np
import torch
from config import config as args
from models.SGF_model import SGF
from modules.MyTrainer import Trainer
from modules.dataloaders import MyDataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer


def main():
    # fix random seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    tokenizer = Tokenizer(args)

    train_dataloader = MyDataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = MyDataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = MyDataLoader(args, tokenizer, split='test', shuffle=False)

    model = SGF(args, tokenizer)
    criterion = compute_loss
    metrics = compute_scores

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
