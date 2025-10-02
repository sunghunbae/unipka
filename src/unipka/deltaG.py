import torch
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader
from .unimol import UniMolModel
from .confgen import ConformerGen


class MolDataset(Dataset):
    """
    A :class:`MolDataset` class is responsible for interface of molecular dataset.
    """
    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
    

class UniMolFreeEnergy(object):
    def __init__(self, model_path, batch_size=32, remove_hs=False, use_gpu=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = UniMolModel(model_path, output_dim=1, remove_hs=remove_hs).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.params = {'remove_hs': remove_hs}


    def decorate_torch_batch(self, batch):
        """
        Prepares a standard PyTorch batch of data for processing by the model. 
        Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input = {k: v.to(self.device) for k, v in net_input.items()}
            net_target = net_target.to(self.device)
        else:
            net_input = {'net_input': net_input.to(self.device)}
            net_target = net_target.to(self.device)
        net_target = None
        return net_input, net_target
    

    def predict(self, smiles_list):
        unimol_input = ConformerGen(**self.params).transform(smiles_list)
        dataset = MolDataset(unimol_input)
        dataloader = DataLoader(dataset, 
                                batch_size=self.batch_size, 
                                shuffle=False,
                                collate_fn=self.model.batch_collate_fn,
                                )

        results = {}
        batch_idx = 0
        for batch in dataloader:
            batch_idx += 1
            i = self.batch_size * (batch_idx-1) # first index
            j = max(self.batch_size * batch_idx, len(smiles_list)) # last index + 1
            net_input, _ = self.decorate_torch_batch(batch)
            with torch.no_grad():
                predictions = self.model(**net_input)
                for smiles, energy in zip(smiles_list[i:j], predictions):
                    results[smiles] = energy.item()
        return results