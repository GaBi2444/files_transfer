import sys
sys.path.append('../')
from reason.options.train_options import TrainOptions
from reason.datasets import get_dataloader
from reason.models.parser import Seq2seqParser
from reason.trainer import Trainer
import pdb


opt = TrainOptions().parse()
train_loader = get_dataloader(opt, 'train')
val_loader = get_dataloader(opt, 'val')
model = Seq2seqParser(opt)
trainer = Trainer(opt, train_loader, val_loader, model)
trainer.train()
