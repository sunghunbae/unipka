from pathlib import Path
# from unipka.ensemble import get_ensemble, read_template, prot, enumerate_template
from unipka.ensemble import MicrostateEnumerator
from unipka.deltaG import FreeEnergyPredictor, calc_distribution
from rdkit import Chem

import importlib
import math


def test_microstate_enumerator():
    microstates = MicrostateEnumerator(smiles="CC(N)C(=O)O")
    E = microstates.ensemble()
    print(E)



if __name__ == "__main__":

    checkpoint_path = Path(__file__).resolve().parent / "../model/t_dwar_v_novartis_a_b.pt"
    # template_a2b_simple, template_b2a_simple = read_template(checkpoint_path / "simple_smarts_pattern.tsv")
    # template_a2b_full, template_b2a_full = read_template(checkpoint_path / "smarts_pattern.tsv")
    pattern_path = importlib.resources.files("unipka.pattern")

    print("checkpoint_dir:", checkpoint_path)
    print("pattern_dir:", pattern_path)    

    predictor = FreeEnergyPredictor(checkpoint_path)
    template_a2b, template_b2a = read_template(pattern_path / "smarts_pattern.tsv")
    
    smi = "NCC(=O)O"
    mol = Chem.MolFromSmiles(smi)

    mol_deprot = Chem.RemoveHs(prot(mol, 4, "a2b"))
    smi_deprot = Chem.MolToSmiles(mol_deprot)

    mol_prot = Chem.RemoveHs(prot(mol, 0, "b2a"))
    smi_prot = Chem.MolToSmiles(mol_prot)

    mol_zwitter = Chem.RemoveHs(prot(mol_prot, 4, "a2b"))
    smi_zwitter = Chem.MolToSmiles(mol_zwitter)

    LN10 = math.log(10)
    TRANSLATE_PH = 6.504894871171601

    print("\nGlycine micro-pKa example:")
    
    print("\nat index = 0 (NH3+ terminal):")
    micro_pKa_1 = predictor.micro_pKa(smi_prot, 0, "a2b")
    print(f"1-H2A(+1) → 1-HA, pKa1: {micro_pKa_1:.2f}")
    
    print("\nat index = 4 (COO- terminal):")
    micro_pKa_2 = predictor.micro_pKa(smi, 4, "a2b")
    print(f"1-HA → 1-A(-1), pKa2: {micro_pKa_2:.2f}")
    
    print("\nat index = 4 (COO- terminal):")
    micro_pKa_3 = predictor.micro_pKa(smi_prot, 4, "a2b")
    print(f"1-H2A(+1) → 2-HA(zwitter ion), pKa3: {micro_pKa_3:.2f}")
    
    micro_pKa_4 = predictor.micro_pKa(smi_zwitter, 0, "a2b")
    print(f"2-HA(zwitter ion) → 1-A(-1), pKa4: {micro_pKa_4:.2f}")
    print(f"pKa1 + pKa2 = {micro_pKa_1:.2f} + {micro_pKa_2:.2f} = {micro_pKa_1 + micro_pKa_2:.2f}")
    print(f"pKa3 + pKa4 = {micro_pKa_3:.2f} + {micro_pKa_4:.2f} = {micro_pKa_3 + micro_pKa_4:.2f}")

    print()
    print("Glutamate macro-pKa example:")
    smi_GLU = "NC(CCC(=O)O)C(=O)O"
    macrostate_A, macrostate_B = enumerate_template(smi_GLU, template_a2b, template_b2a, mode="a2b")
    macrostate_AA, _ = enumerate_template(macrostate_A, template_a2b, template_b2a, mode="b2a")
    _, macrostate_BB = enumerate_template(macrostate_B, template_a2b, template_b2a, mode="a2b")
    
    macro_pKa_1 = predictor.macro_pKa(smi_GLU, template_a2b, template_b2a, "b2a")
    print(f"H3A(+1) → H2A, pKa1: {macro_pKa_1:.2f}")
    
    macro_pKa_2 = predictor.macro_pKa(smi_GLU, template_a2b, template_b2a, "a2b")
    print(f"H2A → HA(-1), pKa2: {macro_pKa_2:.2f}")
    
    macro_pKa_3 = predictor.macro_pKa(macrostate_B, template_a2b, template_b2a, "a2b")
    print(f"HA(-1) → A(-2), pKa2: {macro_pKa_3:.2f}")

    GLU_ensemble = get_ensemble(smi_GLU, template_a2b, template_b2a)
    # Glutamate ensemble: {
    # 0: ['[NH3+]C(CCC(=O)[O-])C(=O)O', '[NH3+]C(CCC(=O)O)C(=O)[O-]', 'NC(CCC(=O)O)C(=O)O'], 
    # -1: ['NC(CCC(=O)O)C(=O)[O-]', '[NH3+]C(CCC(=O)[O-])C(=O)[O-]', 'NC(CCC(=O)[O-])C(=O)O'], 
    # -2: ['NC(CCC(=O)[O-])C(=O)[O-]'], 
    # 1: ['NC(CCC(=O)O)C(O)=[OH+]', 'NC(CCC(O)=[OH+])C(=O)O', '[NH3+]C(CCC(=O)O)C(=O)O'], 
    # 2: ['NC(CCC(O)=[OH+])C(O)=[OH+]', '[NH3+]C(CCC(O)=[OH+])C(=O)O', '[NH3+]C(CCC(=O)O)C(O)=[OH+]'], 
    # 3: ['[NH3+]C(CCC(O)=[OH+])C(O)=[OH+]']}

    GLU_ensemble_free_energy = predictor.ensemble_free_energy(GLU_ensemble)
    # Glutamate ensemble free energy: {
    # 0: [
    # ('[NH3+]C(CCC(=O)[O-])C(=O)O', -1.8589212894439697), 
    # ('[NH3+]C(CCC(=O)O)C(=O)[O-]', -9.405418395996094), 
    # ('NC(CCC(=O)O)C(=O)O', 0.678076446056366)], 
    # -1: [('NC(CCC(=O)O)C(=O)[O-]', -5.9671220779418945), 
    # ('[NH3+]C(CCC(=O)[O-])C(=O)[O-]', -13.925822257995605), 
    # ('NC(CCC(=O)[O-])C(=O)O', -6.114152908325195)], 
    # -2: [('NC(CCC(=O)[O-])C(=O)[O-]', -7.749938011169434)], 
    # 1: [('NC(CCC(=O)O)C(O)=[OH+]', 5.279321670532227), 
    # ('NC(CCC(O)=[OH+])C(=O)O', 5.19708776473999), 
    # ('[NH3+]C(CCC(=O)O)C(=O)O', 1.0517076253890991)], 
    # 2: [('NC(CCC(O)=[OH+])C(O)=[OH+]', 21.181228637695312), 
    # ('[NH3+]C(CCC(O)=[OH+])C(=O)O', 20.87151527404785), 
    # ('[NH3+]C(CCC(=O)O)C(O)=[OH+]', 21.258304595947266)], 
    # 3: [('[NH3+]C(CCC(O)=[OH+])C(O)=[OH+]', 21.352088928222656)]}

    print(f"\nGlutamate ensemble: {GLU_ensemble}")
    print(f"\nGlutamate ensemble free energy: {GLU_ensemble_free_energy}")
    
    # draw_distribution_pH(GLU_ensemble_free_energy)
    print("\nGlutamate distribution at pH 7.4:")
    distribution = calc_distribution(GLU_ensemble_free_energy, 7.4)
    print(distribution)
    # Glutamate distribution at pH 7.4:
    # {0: [('[NH3+]C(CCC(=O)O)C(=O)[O-]', 0.0013606931513400645), 
    # ('[NH3+]C(CCC(=O)[O-])C(=O)O', 7.183863902048043e-07), 
    # ('NC(CCC(=O)O)C(=O)O', 5.682690856709639e-08)], 
    # -1: [('[NH3+]C(CCC(=O)[O-])C(=O)[O-]', 0.9818650038816393), 
    # ('NC(CCC(=O)O)C(=O)[O-]', 0.0003432669589456202), 
    # ('NC(CCC(=O)[O-])C(=O)O', 0.0003976370909890677)], 
    # -2: [('NC(CCC(=O)[O-])C(=O)[O-]', 0.016032618572821056)], 
    # 1: [('NC(CCC(=O)O)C(O)=[OH+]', 7.2636262756194e-11), 
    # ('NC(CCC(O)=[OH+])C(=O)O', 7.886189751045621e-11), 
    # ('[NH3+]C(CCC(=O)O)C(=O)O', 4.9794679615020835e-09)], 
    # 2: [('[NH3+]C(CCC(O)=[OH+])C(=O)O', 1.564752116795825e-18), 
    # ('NC(CCC(O)=[OH+])C(O)=[OH+]', 1.147989496685612e-18), 
    # ('[NH3+]C(CCC(=O)O)C(O)=[OH+]', 1.0628310931699142e-18)], 
    # 3: [('[NH3+]C(CCC(O)=[OH+])C(O)=[OH+]', 1.232052188230532e-19)]}

    print("\nGlutamate distribution at pH 1.2:")
    distribution = calc_distribution(GLU_ensemble_free_energy, 1.2)
    print(distribution)
    # Glutamate distribution at pH 1.2:
    # {0: [('[NH3+]C(CCC(=O)O)C(=O)[O-]', 0.002721427580941209), 
    # ('[NH3+]C(CCC(=O)[O-])C(=O)O', 1.4367945735236121e-06), 
    # ('NC(CCC(=O)O)C(=O)O', 1.136555410467192e-07)], 
    # -1: [('[NH3+]C(CCC(=O)[O-])C(=O)[O-]', 1.2390486647040538e-06), 
    # ('NC(CCC(=O)O)C(=O)[O-]', 4.3318018814922905e-10), 
    # ('NC(CCC(=O)[O-])C(=O)O', 5.017916971060526e-10)], 
    # -2: [('NC(CCC(=O)[O-])C(=O)[O-]', 1.276559446006183e-14)], 
    # 1: [('NC(CCC(=O)O)C(O)=[OH+]', 0.00023024492906148153), 
    # ('NC(CCC(O)=[OH+])C(=O)O', 0.00024997916066930967), 
    # ('[NH3+]C(CCC(=O)O)C(=O)O', 0.015784089159545864)], 
    # 2: [('[NH3+]C(CCC(O)=[OH+])C(=O)O', 7.861078457530611e-06), 
    # ('NC(CCC(O)=[OH+])C(O)=[OH+]', 5.767325958533415e-06), 
    # ('[NH3+]C(CCC(=O)O)C(O)=[OH+]', 5.33950299272988e-06)], 
    # 3: [('[NH3+]C(CCC(O)=[OH+])C(O)=[OH+]', 0.9809925008286093)]}


    for (name, smiles) in [
        ('Cetirizine (Zyrtec) tautomer 1', 'OC(O)=COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1'),
        ('Cetirizine (Zyrtec) tautomer 2', 'O=C(O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1'),
        ('L-lysine','N[C@@H](CCCCN)C(=O)O'),
        ('Guanine', 'C1=NC2=C(N1)C(=O)NC(=N2)N'),
        ]:
        ensemble = get_ensemble(smiles, template_a2b, template_b2a)
        ensemble_free_energy = predictor.ensemble_free_energy(ensemble)
        distribution_74 = calc_distribution(ensemble_free_energy, 7.4)
        distribution_12 = calc_distribution(ensemble_free_energy, 1.2)
        print(f"\nExample {name}:")
        print(f"\nensemble: {ensemble}")
        print(f"\nensemble free energy: {ensemble_free_energy}")
        print(f"\ndistribution at pH 7.4: {distribution_74}")
        print(f"\ndistribution at pH 1.2: {distribution_12}")