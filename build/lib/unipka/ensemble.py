from collections import OrderedDict
from typing import Callable
from dataclasses import dataclass
import importlib
import itertools
import pandas as pd

from rdkit import Chem

from .tautomerism import ComprehensiveTautomers, RdkTautomers


@dataclass
class State:
    smiles: str
    rdmol: Chem.Mol
    protons: dict[int, int] | None = None



# Unreasonable chemical structures to be filtered
DROP_PATTERNS = list(map(Chem.MolFromSmarts, [
    "[#6X5]",
    "[#7X5]",
    "[#8X4]",
    "[*r]=[*r]=[*r]",
    "[#1]-[*+1]~[*-1]",
    "[#1]-[*+1]=,:[*]-,:[*-1]",
    "[#1]-[*+1]-,:[*]=,:[*-1]",
    "[*+2]",
    "[*-2]",
    "[#1]-[#8+1].[#8-1,#7-1,#6-1]",
    "[#1]-[#7+1,#8+1].[#7-1,#6-1]",
    "[#1]-[#8+1].[#8-1,#6-1]",
    "[#1]-[#7+1].[#8-1]-[C](-[C,#1])(-[C,#1])",
    # "[#6;!$([#6]-,:[*]=,:[*]);!$([#6]-,:[#7,#8,#16])]=[C](-[O,N,S]-[#1])",
    # "[#6]-,=[C](-[O,N,S])(-[O,N,S]-[#1])",
    "[OX1]=[C]-[OH2+1]",
    "[NX1,NX2H1,NX3H2]=[C]-[O]-[H]",
    "[#6-1]=[*]-[*]",
    "[cX2-1]",
    "[N+1](=O)-[O]-[H]"
]))




def cnt_stereo_atom(smi: str) -> int:
    """
    Count the stereo atoms in a SMILES
    """
    mol = Chem.MolFromSmiles(smi)
    return sum([str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()])


def stereo_filter(smis: list[str]) -> list[str]:
    """
    A filter against SMILES losing stereochemical information in structure processing.
    """
    filtered_smi_dict = dict()
    for smi in smis:
        nonstereo_smi = Chem.CanonSmiles(smi, useChiral=0)
        stereo_cnt = cnt_stereo_atom(smi)
        if nonstereo_smi not in filtered_smi_dict:
            filtered_smi_dict[nonstereo_smi] = (smi, stereo_cnt)
        else:
            if stereo_cnt > filtered_smi_dict[nonstereo_smi][1]:
                filtered_smi_dict[nonstereo_smi] = (smi, stereo_cnt)
    return [value[0] for value in filtered_smi_dict.values()]


def prot(mol: Chem.Mol, idx: int, mode: str, protons: dict[int, int]) -> State: 
    """Protonate or deprotonate a molecule at a specific site.

    Args:
        mol (Chem.Mol): molecule to be (de)protonated
        idx (int): atom index of reaction site
        mode (str): 'a2b' (protonate) or 'b2a' (deprotonate)

    Returns:
        Chem.Mol: (de)protonated molecule
    """
    mw = Chem.RWMol(mol)
    atom = mw.GetAtomWithIdx(idx)
    molH = None
    xidx : int = idx
    numH = None

    if mode == "a2b":
        if atom.GetAtomicNum() == 1:
            atom_X = atom.GetNeighbors()[0] # only one
            charge = atom_X.GetFormalCharge() -1
            atom_X.SetFormalCharge(charge)
            mw.RemoveAtom(idx) # H atom with the idx is removed
            molH = mw.GetMol()
            xidx = atom_X.GetAtomIdx()
        else:
            charge = atom.GetFormalCharge() -1
            numH = atom.GetTotalNumHs() -1
            atom.SetFormalCharge(charge)
            if numH >= 0:
                atom.SetNumExplicitHs(numH)
            atom.UpdatePropertyCache()
            molH = Chem.AddHs(mw)
    
    elif mode == "b2a":
        charge = atom.GetFormalCharge() + 1
        numH = atom.GetNumExplicitHs() + 1
        atom.SetFormalCharge(charge)
        atom.SetNumExplicitHs(numH)
        molH = Chem.AddHs(mw)

    Chem.SanitizeMol(molH)
    
    # molH = Chem.MolFromSmiles(Chem.MolToSmiles(molH, canonical=False))
    # molH = Chem.AddHs(molH)
    molH_smiles = Chem.CanonSmiles(Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molH))))

    # return (molH, molH_smiles)
    curr_protons = {k:v for k,v in protons.items()}
    curr_protons[xidx] = numH

    return State(smiles=molH_smiles, rdmol=molH, protons=curr_protons)





class Microstate:
    
    def __init__(self, tautomerism: str | None = None, pattern_file: str | None = None, maxiter: int = 10) -> None:
        self.tautomerism = tautomerism
        self.pattern_file = pattern_file
        self.maxiter = maxiter
        self.template_a2b = None
        self.template_b2a = None
        if self.pattern_file is None:
            self._read_template(importlib.resources.files("unipka.rules") / "smarts_pattern.tsv")
        else:
            self._read_template(pattern_file)
        

    def _drop_unreasonable(self, states: list[State]) -> list[State]:
        mask = [] # select mask
        for state in states:
            select = True
            if state.rdmol is None:
                select = False
            else:
                for pattern in DROP_PATTERNS:
                    if len(state.rdmol.GetSubstructMatches(pattern)) > 0:
                        select = False
                        break
            mask.append(select)
        return list(itertools.compress(states, mask))
    

    def _drop_duplicate(self, states: list[State]) -> list[State]:
        U = []
        mask = []
        for state in states:
            if state.smiles in U:
                mask.append(False)
            else:
                mask.append(True)
                U.append(state.smiles)
        return list(itertools.compress(states, mask))


    def _read_template(self, template_file: str) -> None:
        '''
        Read a protonation template.

        Params:
        ----
        `template_file`: path of `.csv`-like template, with columns of substructure names, SMARTS patterns, protonation indices and acid/base flags

        Return:
        ----
        `template_a2b`, `template_b2a`: acid to base and base to acid templates
        '''
        template = pd.read_csv(template_file, sep="\t")

        self.template_a2b = template[template.Acid_or_base == "A"]
        self.template_b2a = template[template.Acid_or_base == "B"]

 
    def _match_template(self, mol: Chem.Mol, mode: str) -> list[int]:
        """Find protonation site using templates

        Args:
            mol (Chem.Mol): input molecule
            mode (str): 'a2b' or 'b2a' for deprotonation or protonation

        Returns:
            list: a list of matched indices to be (de)protonated
        """
        mol = Chem.AddHs(mol)
        template = None
        
        if mode == 'a2b':
            template = self.template_a2b
        elif mode == 'b2a':
            template = self.template_b2a

        matches = []
        for idx, name, smarts, index, acid_base in template.itertuples():
            pattern = Chem.MolFromSmarts(smarts)
            match = mol.GetSubstructMatches(pattern)
            if len(match) == 0:
                continue
            else:
                index = int(index)
                for m in match:
                    matches.append(m[index])
        
        return list(set(matches))


    def _protonate_template(self, smi: str, mode: str) -> list[State]:
        """
        Protonate / Deprotonate a SMILES at every found site in the template

        Params:
        ----
        `template`: `pandas.Dataframe` with columns of substructure names, SMARTS patterns, protonation indices and acid/base flags

        `smi`: The SMILES to be processed

        `mode`: 
            `a2b` means deprotonization, with a hydrogen atom or a heavy atom at `idx`; 
            `b2a` means protonization, with a heavy atom at `idx`
        """
        mol = Chem.MolFromSmiles(smi)
        sites = self._match_template(mol, mode)
        protons = {}
        for site in sites:
            atom = mol.GetAtomWithIdx(site)
            if atom.GetAtomicNum() == 1:
                protons[site] = atom.GetNeighbors()[0].GetNumExplicitHs()
            else:
                protons[site] = atom.GetNumExplicitHs()
        return [prot(mol, site, mode, protons) for site in sites]


    def del_proton(self, smi: str) -> list[State]:
        return self._protonate_template(smi, "a2b")
    

    def add_proton(self, smi: str) -> list[State]:
        return self._protonate_template(smi, "b2a")
    

    def enumerate_tautomers(self, smiles: str) -> list[State]:
        # Moltaut 
        # ct = ComprehensiveTautomers(smiles).enumerate()
        if self.tautomerism == "comprehensive":
            t = ComprehensiveTautomers(smiles).enumerate()
        elif self.tautomerism == "rdkit":
            t = RdkTautomers(smiles).enumerate()
        else:
            return [State(smiles=s, rdmol=Chem.MolFromSmiles(s)) for s in smiles]
        
        return [State(smiles=s, rdmol=Chem.MolFromSmiles(s)) for s in t.enumerated]


    def enumerate_template(self, init_states: list[State], mode: str) -> tuple[list[State], list[State]]:
        """
        Enumerate all the (de)protonation results of one SMILES.

        Params:
        ----
        `smi`: The smiles to be processed.

        `template_a2b`: `pandas.Dataframe` with columns of substructure names, SMARTS patterns, deprotonation indices and acid flags.

        `template_b2a`: `pandas.Dataframe` with columns of substructure names, SMARTS patterns, protonation indices and base flags.

        `mode`: 
            - "a2b": `smi` is an acid to be deprotonated.
            - "b2a": `smi` is a base to be protonated.

        `maxiter`: Max iteration number of template matching and microstate pool growth.

        `verbose`:
            - 0: Silent mode.
            - 1: Print the length of microstate pools in each iteration.
            - 2: Print the content of microstate pools in each iteration.

        `filter_patterns`: Unreasonable chemical structures.

        Return:
        ----
        A microstate pool and B microstate pool after enumeration.
        """
        pool_A = []
        pool_B = []

        # if isinstance(smiles, str):
        #     init_states = [State(smiles=smiles, rdmol=Chem.MolFromSmiles(smiles))]
        # if isinstance(smiles, list):
        #     init_states = [State(smiles=s, rdmol=Chem.MolFromSmiles(s)) for s in smiles]
         
        if mode == "a2b":
            pool_A = init_states
        elif mode == "b2a":
            pool_B = init_states
        
        pool_A_size = -1
        pool_B_size = -1
        i = 0

        # coupled enumeration
        while (len(pool_A) != pool_A_size or len(pool_B) != pool_B_size) and i < self.maxiter:
            pool_A_size, pool_B_size = len(pool_A), len(pool_B)
            if (mode == "a2b" and (i + 1) % 2) or (mode == "b2a" and i % 2):
                smis_A_tmp_pool = []
                for st in pool_A:
                    pool_B += self._drop_unreasonable(self.del_proton(st.smiles))
                    smis_A_tmp_pool += self._drop_unreasonable(self.enumerate_tautomers(st.smiles))
                pool_A += smis_A_tmp_pool
            elif (mode == "b2a" and (i + 1) % 2) or (mode == "a2b" and i % 2):
                smis_B_tmp_pool = []
                for st in pool_B:
                    pool_A += self._drop_unreasonable(self.add_proton(st.smiles))
                    smis_B_tmp_pool += self._drop_unreasonable(self.enumerate_tautomers(st.smiles))
                pool_B += smis_B_tmp_pool
            
            pool_A = self._drop_unreasonable(pool_A)
            pool_B = self._drop_unreasonable(pool_B)
            pool_A = self._drop_duplicate(pool_A)
            pool_B = self._drop_duplicate(pool_B)
            i += 1
        
        return pool_A, pool_B


    def ensemble(self, smiles: str) -> dict[int, list[str]]:
        """Get the protonation state ensemble of a SMILES.

        Args:
            smi (str): SMILES
            template_a2b (pd.DataFrame): transition from acid to base
            template_b2a (pd.DataFrame): trasnsition from base to acid
            maxiter (int, optional): max iterations. Defaults to 10.

        Returns:
            dict[int, list[str]]: {formal charge: [SMILES]}
        """
        _ensemble = dict()
        q0 = Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))
        _ensemble[q0] = [State(smiles=smiles, rdmol=Chem.MolFromSmiles(smiles))]

        pool_0 = [State(smiles=smiles, rdmol=Chem.MolFromSmiles(smiles))]
        pool_0, pool_b1 = self.enumerate_template(pool_0, mode="a2b")
        
        if pool_b1:
            _ensemble[q0 - 1] = pool_b1
        
        q = q0 - 2
        while True:
            if q + 1 in _ensemble:
                _, pool_b = self.enumerate_template(_ensemble[q + 1], mode="a2b")
                if pool_b:
                    _ensemble[q] = pool_b
                else:
                    break
            q -= 1

        pool_a1, pool_0 = self.enumerate_template(pool_0, mode="b2a")
        if pool_a1:
            _ensemble[q0 + 1] = pool_a1
        q = q0 + 2
        while True:
            if q - 1 in _ensemble:
                pool_a, _ = self.enumerate_template(_ensemble[q - 1], mode="b2a")
                if pool_a:
                    _ensemble[q] = pool_a
                else:
                    break
            q += 1
        
        _ensemble[q0] = pool_0

        unique_ensemble = {}
        for k, v in _ensemble.items():
            unique_ensemble[k] = self._drop_duplicate(v)
        
        return unique_ensemble