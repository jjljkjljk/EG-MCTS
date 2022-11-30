import numpy as np
import torch
import random
import logging
from eg_mcts.arg.parse_args import args
from eg_mcts.model.eg_network import EG_MLP
from eg_mcts.model.train_data_loader import ValueDataLoader
from eg_mcts.model.train_method import Trainer
from eg_mcts.utils import setup_logger

def train():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    model = EG_MLP(
        n_layers=args.n_layers,
        fp_dim=args.value_fp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=0.1,
        device=device
    )
    model_f = '%s/%s' % (args.save_folder, args.value_model)
    logging.info('Loading Experience Guidance Network from %s' % model_f)
    model.load_state_dict(torch.load(model_f, map_location=device))
    train_data_loader = ValueDataLoader(
        fp_value_f='%s/%s' % (args.train_root, args.train_data),
        batch_size=args.batch_size
    )


    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        save_epoch_int=args.save_epoch_int,
        model_folder=args.save_folder,
        device=device
    )

    trainer.train()


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('train.log')
    train()
