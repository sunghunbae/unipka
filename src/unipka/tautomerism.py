"""
Reference:

D. K. Dhaked, W.-D. Ihlenfeldt, H. Patel, V. Delannée, M. C. Nicklaus, 
Toward a Comprehensive Treatment of Tautomerism in Chemoinformatics Including in InChI V2. 
J. Chem. Inf. Model. (2020). https://doi.org/10.1021/acs.jcim.9b01080.

https://github.com/xundrug/moltaut

"""

from importlib.resources import files, as_file

from collections import defaultdict

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')


class RdkTautomers:
    """RDKit built-in tautomer enumeration
    
    M. Sitzmann, W.-D. Ihlenfeldt, M. C. Nicklaus, Tautomerism in large databases. 
    J. Comput. Aided Mol. Des. 24, 521-551 (2010).
    """
    def __init__(self, smiles: str):
        self.smiles : str = smiles
        self.rdmol : Chem.Mol = Chem.MolFromSmiles(smiles)
        self.enumerator = rdMolStandardize.TautomerEnumerator()

    def enumerate(self):
        tautomers = self.enumerator.Enumerate(self.rdmol)
        self.enumerated : list[str] = [Chem.MolToSmiles(t) for t in tautomers]
        return self
    
    def canonicalize(self):
        can_tautomer = self.enumerator.Canonicalize(self.rdmol)
        self.canonical : str = Chem.MolToSmiles(can_tautomer)
        return self
    

class ComprehensiveTautomers:
    """Comprehensive tautomer enumeration based on MolTaut implementation
    D. K. Dhaked, W.-D. Ihlenfeldt, H. Patel, V. Delannée, M. C. Nicklaus,
    Toward a Comprehensive Treatment of Tautomerism in Chemoinformatics Including in InChI V2.
    J. Chem. Inf. Model. (2020). https://doi.org/10.1021/acs.jcim.9b01080.
    """

    ps = Chem.SmilesParserParams()
    ps.removeHs = False

    uncharger = rdMolStandardize.Uncharger()

    forbidden = [ "O=[N+]([O-])", ]
    forbidden_patterns = [Chem.MolFromSmarts(_) for _ in forbidden]

    def __init__(self, smiles: str):
        self.smiles : str = smiles
        self.rdmol : Chem.Mol = Chem.MolFromSmiles(smiles)
        # self.rdmol = self.uncharger.uncharge(self.rdmol)
        self.dict : dict[str,list[str]] = {'self' : [self.smiles]} # transform dictionary
        self.smirks : list[tuple[str,str]] = []
        self.revdict = defaultdict(list) # reverse dictionary
        # SMIRKS (SMILES ReaKtion Specification)
        with as_file(files('unipka.rules') / "smirks_transform_all.txt") as path:
            with open(path, 'r') as f:
                # ex. [#1:1][C0:2]#[N0:3]>>[C-:2]#[N+:3][#1:1]	PT_20_00
                contents = f.readlines()
                self.smirks = [line.strip().split("\t") for line in contents]
                # initialize
                for idx, (smrk, name) in enumerate(self.smirks):
                    self.dict[str(idx) + "_" + name] = []
        
        self.enumerated : list[str] = [] # enumerated tautomers (SMILES)
        self.num_confs = None
        self.optimizer = None
        self.device = None
        self.nmodel = None
        self.imodel = None
        self.ordered : list = []
        self.popular : list = []
        self.states : list[tuple[str, str]] = []
        self.microstates : list = []


    def _contains_phosphorus(self) -> bool:
        return any([at.GetAtomicNum() == 15 for at in self.rdmol.GetAtoms()])


    def _acceptable(self, rom: Chem.Mol) -> bool:
        if not rom:
            return False
        # more robust way to check substructure match 
        # than using Chem.Mol.HasSubstructMatch
        for pattern in self.forbidden_patterns:
            matches = sum(rom.GetSubstructMatches(pattern), ())
            for at in rom.GetAtoms():
                if (at.GetFormalCharge() != 0) and (at.GetIdx() not in matches):
                    return False
        return True 


    def _kekulized(self, rwm: Chem.Mol) -> list[Chem.Mol]:
        rwm = Chem.AddHs(rwm)
        mols = Chem.ResonanceMolSupplier(rwm, Chem.KEKULE_ALL)
        # _kekulized = [_ for _ in mols if self._acceptable(_)]
        _kekulized = []
        for _ in mols:
            if not self._acceptable(_):
                continue
            try:
                smi = Chem.MolToSmiles(_, kekuleSmiles=True)
            except:
                print("Kekulize error:", Chem.MolToSmiles(_))
                continue
            _kekulized.append(Chem.MolFromSmiles(smi, ComprehensiveTautomers.ps))
        return _kekulized


    def _transform(self, rwm: Chem.Mol, protect_dummy: bool = True):
        if protect_dummy:
            # mark dummy atoms for protection
            # RDKit recognizes the special property key _protected. 
            # If this property is set to a "truthy" value (like '1'), 
            # RDKit will ignore this atom when matching the reactant patterns 
            # of the chemical reaction.
            for atom in rwm.GetAtoms():
                if atom.GetAtomicNum() == 0: # if atom == "*": # dummy atom
                    atom.SetProp('_protected', '1')

        # Kekule form of the SMILES
        Chem.Kekulize(rwm, clearAromaticFlags=True)

        for idx, (smrk, name) in enumerate(self.smirks):
            rxn = AllChem.ReactionFromSmarts(smrk)
            new_molecules = rxn.RunReactants((rwm,))
            if len(new_molecules) > 0:
                for unit in new_molecules:
                    _ = Chem.MolToSmiles(unit[0])
                    _mol = Chem.MolFromSmiles(_, sanitize=True)
                    _smi = None
                    if _mol:
                        _smi = Chem.MolToSmiles(_mol)
                    if _smi:
                        self.dict[str(idx)+"_"+name].append(_smi)


    def _collect(self):
        for rule, smis in self.dict.items():
            for _ in set(smis):
                self.revdict[_].append(rule)
        return [smi for smi, rules in self.revdict.items()]


    def enumerate(self):
        m = Chem.Mol(self.rdmol) # copy
        if self._contains_phosphorus():
            self.enumarated = [self.smiles]
        else:
            kms = self._kekulized(m)  
            for km in kms: # can be parallel
                self._transform(km)

            for i in range(5):
                transformed_smis = []
                for rule, smis in self.dict.items():
                    transformed_smis += smis # list
                transformed_smis = set(transformed_smis)
                transformed_mols = [Chem.MolFromSmiles(_) for _ in transformed_smis]
                for tm in transformed_mols: # can be parallel
                    kms = self._kekulized(tm)
                    for km in kms:
                        self._transform(km)
      
            self.enumerated = self._collect()
        
        return self