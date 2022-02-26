import os
import numpy as np
import torch
import pickle
import logging
from torch.utils.data import Dataset, DataLoader

# experience [mol[:],template[:],Q[:]]
# mol_fp 1024 + template_fp 1024




class ValueDataset(Dataset):
    def __init__(self, fp_value_f):
        assert os.path.exists('%s.pkl' % fp_value_f)
        logging.info('Loading value dataset from %s.pkl'% fp_value_f)

        data_dict = pickle.load(open('%s.pkl' % fp_value_f, 'rb'))


        self.mol = data_dict['mol']
        self.template = data_dict['template']
        self.Q_value = torch.Tensor(data_dict['Q_value']).reshape((-1, 1))
        self.reshuffle()

        logging.info('%d ((m,T), value) pairs loaded' % self.Q_value.shape[0])

        logging.info(
            'mean: %f, std:%f, min: %f, max: %f, zeros: %f' %
            (self.Q_value.mean(), self.Q_value.std(), self.Q_value.min(),
             self.Q_value.max(), (self.Q_value==0).sum()*1. / self.Q_value.shape[0])
        )

    def reshuffle(self):
        shuffle_idx = np.random.permutation(self.Q_value.shape[0])
        self.Q_value = self.Q_value[shuffle_idx]
        self.mol = np.array(self.mol)[shuffle_idx]
        self.template = np.array(self.template)[shuffle_idx]


    def __len__(self):
        return self.Q_value.shape[0]

    def __getitem__(self, index):
        return self.mol[index], self.template[index],self.Q_value[index]


class ValueDataLoader(DataLoader):
    def __init__(self, fp_value_f, batch_size, shuffle=False):
        self.dataset = ValueDataset(fp_value_f)

        super(ValueDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def reshuffle(self):
        self.dataset.reshuffle()

