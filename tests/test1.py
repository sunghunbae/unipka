from pathlib import Path
from unipka.ensemble import Microstate
from unipka.deltaG import FreeEnergyPredictor, calc_distribution

import pytest


@pytest.fixture
def test_ensemble():
    # microstate = Microstate(tautomerism="comprehensive")
    microstate = Microstate()
    # E = microstate.ensemble(smiles="NC(CCC(=O)O)C(=O)O") # glutamate
    # E = microstate.ensemble(smiles="C1=NC2=C(N1)C(=O)NC(=N2)N") # guanine
    E = microstate.ensemble(smiles='OC(O)=COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1') # cetirizine
    # for (name, smiles) in [
    #     ('Cetirizine (Zyrtec) tautomer 1', 'OC(O)=COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1'),
    #     ('Cetirizine (Zyrtec) tautomer 2', 'O=C(O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1'),
    #     ('L-lysine','N[C@@H](CCCCN)C(=O)O'),
    #     ('Guanine', 'C1=NC2=C(N1)C(=O)NC(=N2)N'),
    print(E)
    return E


@pytest.fixture
def test_free_energy(test_ensemble):
    E = test_ensemble
    predictor = FreeEnergyPredictor(Path(__file__).resolve().parent / "../model/t_dwar_v_novartis_a_b.pt")
    ensemble_free_energy = predictor.ensemble_free_energy(E)
    print(ensemble_free_energy)
    return ensemble_free_energy


def test_distribution_74(test_free_energy):
    ensemble_free_energy = test_free_energy
    distribution = calc_distribution(ensemble_free_energy, 7.4)
    print("pH 7.4 distribution")
    for k, v in distribution.items():
        print(k)
        for (smiles, prob) in sorted(v, key=lambda x: x[1], reverse=True):
            if prob > 0.1:
                print(smiles, prob)


def test_distribution_12(test_free_energy):
    ensemble_free_energy = test_free_energy
    distribution = calc_distribution(ensemble_free_energy, 1.2)
    print("pH 1.2 distribution")
    for k, v in distribution.items():
        print(k)
        for (smiles, prob) in sorted(v, key=lambda x: x[1], reverse=True):
            if prob > 0.1:
                print(smiles, prob)