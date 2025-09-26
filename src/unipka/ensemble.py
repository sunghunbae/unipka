from collections import OrderedDict
from typing import Callable
import importlib
import pandas as pd

from rdkit import Chem

from .tautomerism import ComprehensiveTautomers, RdkTautomers


# Unreasonable chemical structures to be filtered
FILTER_PATTERNS = list(map(Chem.MolFromSmarts, [
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


def sanitize_checker(smi: str, filter_patterns: list[Chem.Mol], verbose: bool=False) -> bool:
    """
    Check if a SMILES can be sanitized and does not contain unreasonable chemical structures.

    Params:
    ----
    `smi`: The SMILES to be check.

    `filter_patterns`: Unreasonable chemical structures.

    `verbose`: If True, matched unreasonable chemical structures will be printed.

    Return:
    ----
    If the SMILES should be filtered.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    for pattern in filter_patterns:
        match = mol.GetSubstructMatches(pattern)
        if match:
            if verbose:
                print(f"pattern {pattern}")
            return False
    try:
        Chem.SanitizeMol(mol)
    except:
        print("cannot sanitize")
        return False
    return True


def sanitize_filter(smis: list[str], filter_patterns: list[Chem.Mol]=FILTER_PATTERNS) -> list[str]:
    """
    A filter for SMILES can be sanitized and does not contain unreasonable chemical structures.

    Params:
    ----
    `smis`: The list of SMILES.

    `filter_patterns`: Unreasonable chemical structures.

    Return:
    ----
    The list of SMILES filtered.
    """
    def _checker(smi):
        return sanitize_checker(smi, filter_patterns)
    return list(filter(_checker, smis))


def make_filter(filter_param: OrderedDict) -> Callable:
    """
    Make a sequential SMILES filter

    Params:
    ----
    `filter_param`: An `collections.OrderedDict` whose keys are single filter functions and the corresponding values are their parameter dictionary.

    Return:
    ----
    The sequential filter function
    """
    def seq_filter(smis):
        for single_filter, param in filter_param.items():
            smis = single_filter(smis, **param)
        return smis
    return seq_filter


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


def prot(mol: Chem.Mol, idx: int, mode: str) -> tuple[Chem.Mol, str]: 
    """Protonate or deprotonate a molecule at a specific site.

    Args:
        mol (Chem.Mol): molecule to be (de)protonated
        idx (int): atom index of reaction site
        mode (str): 'a2b' (protonate) or 'b2a' (deprotonate)

    Returns:
        Chem.Mol: (de)protonated molecule
    """
    mw = Chem.RWMol(mol)
    if mode == "a2b":
        atom_H = mw.GetAtomWithIdx(idx)
        if atom_H.GetAtomicNum() == 1:
            atom_A = atom_H.GetNeighbors()[0]
            charge_A = atom_A.GetFormalCharge()
            atom_A.SetFormalCharge(charge_A - 1)
            mw.RemoveAtom(idx)
            mol_prot = mw.GetMol()
        else:
            charge_H = atom_H.GetFormalCharge()
            numH_H = atom_H.GetTotalNumHs()
            atom_H.SetFormalCharge(charge_H - 1)
            atom_H.SetNumExplicitHs(numH_H - 1)
            atom_H.UpdatePropertyCache()
            mol_prot = Chem.AddHs(mw)
    elif mode == "b2a":
        atom_B = mw.GetAtomWithIdx(idx)
        charge_B = atom_B.GetFormalCharge()
        atom_B.SetFormalCharge(charge_B + 1)
        numH_B = atom_B.GetNumExplicitHs()
        atom_B.SetNumExplicitHs(numH_B + 1)
        mol_prot = Chem.AddHs(mw)
    Chem.SanitizeMol(mol_prot)
    mol_prot = Chem.MolFromSmiles(Chem.MolToSmiles(mol_prot, canonical=False))
    mol_prot = Chem.AddHs(mol_prot)
    _ = Chem.Mol(mol_prot) # copy
    mol_prot_smiles = Chem.CanonSmiles(Chem.MolToSmiles(Chem.RemoveHs(_)))

    return (mol_prot, mol_prot_smiles)



class Microstate:
    def __init__(self, tautomerism: str | None = None, pattern_file: str | None = None, maxiter: int = 10) -> None:
        self.tautomerism = tautomerism
        self.pattern_file = pattern_file
        self.maxiter = maxiter
        self.template_a2b = None
        self.template_b2a = None
        if self.pattern_file is None:
            pattern_path = importlib.resources.files("unipka.rules")
            self._read_template(pattern_path / "smarts_pattern.tsv")
        else:
            self._read_template(pattern_file)
        
        self.filters = make_filter({
            sanitize_filter: {"filter_patterns": FILTER_PATTERNS},
            stereo_filter: {}
        })


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


    def _protonate_template(self, smi: str, mode: str) -> tuple[list[int], list[str]]:
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
        smis = []
        for site in sites:
            (protonated_mol, protonated_smi) = prot(mol, site, mode)
            smis.append(protonated_smi)
            # smis.append(Chem.CanonSmiles(Chem.MolToSmiles(Chem.RemoveHs(prot(mol, site, mode)))))
        return sites, list(set(smis))


    def del_proton(self, smi: str) -> list[str]:
        return self._protonate_template(smi, "a2b")[1]
    

    def add_proton(self, smi: str) -> list[str]:
        return self._protonate_template(smi, "b2a")[1]
    

    def enumerate_tautomers(self, smiles: str) -> list[str]:
        # Moltaut 
        # ct = ComprehensiveTautomers(smiles).enumerate()
        if self.tautomerism == "comprehensive":
            t = ComprehensiveTautomers(smiles).enumerate()
        elif self.tautomerism == "rdkit":
            t = RdkTautomers(smiles).enumerate()
        else:
            return [smiles]
        return t.enumerated


    def enumerate_template(self, smiles: str | list[str], mode: str="A") -> tuple[list[str], list[str]]:
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
        if isinstance(smiles, str):
            smis = [smiles]
        else:
            smis = smiles

        if mode == "a2b":
            smis_A_pool, smis_B_pool = smis, []
        elif mode == "b2a":
            smis_A_pool, smis_B_pool = [], smis
        
        pool_length_A = -1
        pool_length_B = -1
        i = 0
        while (len(smis_A_pool) != pool_length_A or len(smis_B_pool) != pool_length_B) and i < self.maxiter:
            pool_length_A, pool_length_B = len(smis_A_pool), len(smis_B_pool)
            if (mode == "a2b" and (i + 1) % 2) or (mode == "b2a" and i % 2):
                smis_A_tmp_pool = []
                for smi in smis_A_pool:
                    smis_B_pool += self.filters(self.del_proton(smi))
                    smis_A_tmp_pool += self.filters(self.enumerate_tautomers(smi))
                smis_A_pool += smis_A_tmp_pool
            elif (mode == "b2a" and (i + 1) % 2) or (mode == "a2b" and i % 2):
                smis_B_tmp_pool = []
                for smi in smis_B_pool:
                    smis_A_pool += self.filters(self.add_proton(smi))
                    smis_B_tmp_pool += self.filters(self.enumerate_tautomers(smi))
                smis_B_pool += smis_B_tmp_pool
            smis_A_pool = self.filters(smis_A_pool)
            smis_B_pool = self.filters(smis_B_pool)
            smis_A_pool = list(set(smis_A_pool))
            smis_B_pool = list(set(smis_B_pool))
            i += 1

        smis_A_pool = list(map(Chem.CanonSmiles, smis_A_pool))
        smis_B_pool = list(map(Chem.CanonSmiles, smis_B_pool))
        
        return smis_A_pool, smis_B_pool


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
        _ensemble[q0] = [smiles]

        smis_0 = [smiles]

        smis_0, smis_b1 = self.enumerate_template(smis_0, mode="a2b")
        if smis_b1:
            _ensemble[q0 - 1] = smis_b1
        q = q0 - 2
        while True:
            if q + 1 in _ensemble:
                _, smis_b = self.enumerate_template(_ensemble[q + 1], mode="a2b")
                if smis_b:
                    _ensemble[q] = smis_b
                else:
                    break
            q -= 1

        smis_a1, smis_0 = self.enumerate_template(smis_0, mode="b2a")
        if smis_a1:
            _ensemble[q0 + 1] = smis_a1
        q = q0 + 2
        while True:
            if q - 1 in _ensemble:
                smis_a, _ = self.enumerate_template(_ensemble[q - 1], mode="b2a")
                if smis_a:
                    _ensemble[q] = smis_a
                else:
                    break
            q += 1
        
        _ensemble[q0] = smis_0

        unique_ensemble = {}
        for k, v in _ensemble.items():
            unique_ensemble[k] = list(set(v))
        
        return unique_ensemble