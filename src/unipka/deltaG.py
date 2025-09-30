
import torch
import numpy as np
import math
import pandas as pd
from typing import Literal
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

from .unimol import UniMolModel
from .confgen import ConformerGen
from .ensemble import Microstate, prot


LN10 = math.log(10)
TRANSLATE_PH = 6.504894871171601

# kT = 0.001987 * 298.0 # (kcal/mol K), standard condition
# C = math.log(10) * kT

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
    

class FreeEnergyPredictor(object):
    def __init__(self, model_path, batch_size=32, remove_hs=False, use_gpu=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = UniMolModel(model_path, output_dim=1, remove_hs=remove_hs).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.params = {'remove_hs': remove_hs}
        self.conformer_gen = ConformerGen(**self.params)

    def preprocess_data(self, smiles_list):
        # conf gen
        inputs = self.conformer_gen.transform(smiles_list)
        return inputs

    def predict(self, smiles_list):
        unimol_input = self.preprocess_data(smiles_list)
        dataset = MolDataset(unimol_input)
        dataloader = DataLoader(dataset, 
                                batch_size=self.batch_size, 
                                shuffle=False,
                                collate_fn=self.model.batch_collate_fn,
                                )

        results = {}
        for batch in dataloader:
            net_input, _ = self.decorate_torch_batch(batch)
            with torch.no_grad():
                predictions = self.model(**net_input)
                for smiles, energy in zip(smiles_list, predictions):
                    results[smiles] = energy.item()
        return results

    def predict_single(self, smiles):
        return self.predict([smiles])

    def decorate_torch_batch(self, batch):
        """
        Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {
                k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {'net_input': net_input.to(
                self.device)}, net_target.to(self.device)
        net_target = None

        return net_input, net_target
    

    def micro_pKa(self, smi: str, idx: int, mode: Literal["a2b", "b2a"]) -> float:
        mol = Chem.MolFromSmiles(smi)
        new_mol = Chem.RemoveHs(prot(mol, idx, mode))
        new_smi = Chem.MolToSmiles(new_mol)
        if mode == "a2b":
            smi_A = smi
            smi_B = new_smi
        elif mode == "b2a":
            smi_B = smi
            smi_A = new_smi
        DfGm = self.predict([smi_A, smi_B])
        pKa = (DfGm[smi_B] - DfGm[smi_A]) / LN10 + TRANSLATE_PH

        return pKa

    @staticmethod    
    def log_sum_exp(DfGm: list[float]) -> float:
        return math.log10(sum([math.exp(-g) for g in DfGm]))


    def macro_pKa(self, smi: str, mode: Literal["a2b", "b2a"]) -> float:
        microstates = Microstate(smiles=smi)
        macrostate_A, macrostate_B = microstates.enumerate_template(mode)
        DfGm_A = self.predict(macrostate_A)
        DfGm_B = self.predict(macrostate_B)

        return self.log_sum_exp(DfGm_A.values()) - self.log_sum_exp(DfGm_B.values()) + TRANSLATE_PH


    def ensemble_free_energy(self, ensemble: dict[int, list[str]]) -> dict[int, tuple[str, float]]:
        ensemble_free_energy = dict()
        for q, macrostate in ensemble.items():
            prediction = self.predict(macrostate)
            ensemble_free_energy[q] = [(microstate, prediction[microstate]) for microstate in macrostate]
        
        return ensemble_free_energy
    

def calc_distribution(ensemble_free_energy: dict[int, dict[str, float]], pH: float) -> dict[int, dict[str, float]]:
    ensemble_boltzmann_factor = defaultdict(list)
    partition_function = 0
    for q, macrostate_free_energy in ensemble_free_energy.items():
        for microstate, DfGm in macrostate_free_energy:
            boltzmann_factor = math.exp(-DfGm - q * LN10 * (pH - TRANSLATE_PH))
            partition_function += boltzmann_factor
            ensemble_boltzmann_factor[q].append((microstate, boltzmann_factor))
    return {
        q: [(microstate, boltzmann_factor / partition_function) for microstate, boltzmann_factor in macrostate_boltzmann_factor] 
        for q, macrostate_boltzmann_factor in ensemble_boltzmann_factor.items()
    }