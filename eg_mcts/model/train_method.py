import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import logging
from eg_mcts.utils.smiles_process import batch_datas_to_fp
def unpack_fps(packed_fps):

    # packed_fps = np.array(packed_fps)
    shape = (*(packed_fps.shape[:-1]), -1)
    # fps = np.unpackbits(packed_fps.reshape((-1, packed_fps.shape[-1])),
    #                    axis=-1)
    fps = torch.FloatTensor(packed_fps).view(shape)

    return fps


class Trainer:
    def __init__(self, model, train_data_loader, n_epochs, lr,
                 save_epoch_int, model_folder, device):
        self.train_data_loader = train_data_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.save_epoch_int = save_epoch_int
        self.model_folder = model_folder
        self.device = device
        self.model = model.to(self.device)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

    def _pass(self, data, train=True):
        self.optim.zero_grad()
        mol, template, values = data
        fps = unpack_fps(batch_datas_to_fp(mol,template))
        v_pred = self.model(fps)
        loss = F.mse_loss(v_pred, values)

        if train:
            loss.backward()
            self.optim.step()

        return loss.item()

    def _train_epoch(self):
        self.model.train()

        losses = []
        pbar = tqdm(self.train_data_loader)
        for data in pbar:
            loss = self._pass(data)
            losses.append(loss)
            pbar.set_description('[loss: %f]' % (loss))

        return np.array(losses).mean()


    def train(self):
        best_val_loss = np.inf
        for epoch in range(self.n_epochs):
            # self.train_data_loader.reshuffle()

            train_loss = self._train_epoch()
            logging.info(
                '[Epoch %d/%d] [training loss: %f]' %
                (epoch, self.n_epochs, train_loss)
            )

            if (epoch + 1) % self.save_epoch_int == 0:
                save_file = self.model_folder + '/epoch_%d.pt' % epoch
                torch.save(self.model.state_dict(), save_file)
