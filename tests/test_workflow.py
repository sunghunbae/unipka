from unipka.deltaG import UniMolFreeEnergy
from rdworks import StateNetwork
import pathlib
import math
import numpy as np

workdir = pathlib.Path(__file__).resolve().parent

LN10 = math.log(10)
TRANSLATE_PH = 6.504894871171601 
# unipka model specific variable for pH dependent deltaG
#./unimol/tasks/unimol_mlm.py:            self.mean = 6.504894871171601  # precompute from dwar_8228 full set
#./unimol/tasks/unimol_pka.py:            self.mean = 6.504894871171601  # precompute from dwar_8228 full set
# free energy might be obtained at pH 6.504...

def test_network():
    #smiles = 'C1(C=CC=C2)=C2C(NCC3=CC=CC=C3)=NC=N1'
    #smiles = 'c1ccc(C[N-]c2ncnc3ccccc23)cc1'
    smiles = 'c1ccc(CNc2ncnc3ccccc23)cc1'

    model_path = workdir / "../model/t_dwar_v_novartis_a_b.pt"
    predictor = UniMolFreeEnergy(model_path)
    sn = StateNetwork()
    sn.build(smiles=smiles, max_formal_charge=3, tautomer_rule=None)

    E = [st.smiles for st in sn.visited_states]
    Q = [st.charge for st in sn.visited_states]
    P7 = []
    P1 = []

    dG = predictor.predict(E) # dictionary
    PE = [dG[st.smiles] for st in sn.visited_states]

    micro_pKa = {}
    for k, v in sn.micro_pKa(PE).items():
        micro_pKa[k] = (np.array(v)/LN10 + TRANSLATE_PH).tolist()

    macro_pKa = (np.array(sn.macro_pKa(PE))/LN10 + TRANSLATE_PH).tolist()

    pop74 = sn.population(PE, pH=(7.4-TRANSLATE_PH), C=LN10, kT=1.0)
    pop12 = sn.population(PE, pH=(1.2-TRANSLATE_PH), C=LN10, kT=1.0)

    print("\n")
    for k, st in enumerate(sn.visited_states):
        print(f"{k:2} {st.smiles:50} {PE[k]:8.3f} {pop12[k]:8.3g} {pop74[k]:8.3g}")

    print(f"\nmacro-pKa: {macro_pKa}")
    print(f"micro-pKa: {micro_pKa}")
    
    # Input (SAMPL6 SM07): c1ccc(CNc2ncnc3ccccc23)cc1
    # experimental macro pKa : 6.08 +/- 0.01

    # QupKake results:
    #     idx= 5 pka_type= basic pka= 4.136
    #     idx= 7 pka_type= basic pka= 4.277
    #     idx= 9 pka_type= basic pka= 4.257
    #     idx= 5 pka_type= acidic pka= 12.598

    # Uni-pKa results:
    #  0 c1ccc(CNc2ncnc3ccccc23)cc1                           -6.025 2.69e-05    0.984
    #  1 c1ccc(C[NH2+]c2ncnc3ccccc23)cc1                      -2.920    0.244  0.00561
    #  2 c1ccc(CNc2[nH+]cnc3ccccc23)cc1                       -2.741    0.204  0.00469
    #  3 c1ccc(CNc2nc[nH+]c3ccccc23)cc1                       -2.964    0.254  0.00587
    #  4 c1ccc(C[N-]c2ncnc3ccccc23)cc1                         7.657 1.53e-16 8.83e-06
    #  5 c1ccc(C[NH2+]c2[nH+]cnc3ccccc23)cc1                  19.674 7.57e-06  1.1e-13
    #  6 c1ccc(C[NH2+]c2nc[nH+]c3ccccc23)cc1                  21.270 1.53e-06 2.23e-14
    #  7 c1ccc(CNc2[nH+]c[nH+]c3ccccc23)cc1                   11.912   0.0178 2.59e-10
    #  8 c1ccc(C[N-]c2[nH+]cnc3ccccc23)cc1                     7.562 3.38e-11 1.24e-06
    #  9 c1ccc(C[N-]c2nc[nH+]c3ccccc23)cc1                    10.144 2.56e-12 9.35e-08
    # 10 c1ccc(C[NH2+]c2[nH+]c[nH+]c3ccccc23)cc1              21.369    0.281 2.57e-15
    # 11 c1ccc(C[N-]c2[nH+]c[nH+]c3ccccc23)cc1                12.133 7.07e-08 1.63e-09
    # macro-pKa: [5.248700101444452, 12.3693201962264]
    # micro-pKa: {5: [5.263056256779453, 12.37074708270436], 7: [5.190629233663213], 9: [5.280577863433882]}

    # Note - UnipKa is not able to differentiate two tautomers(*)
    # Input: C1(C=CC=C2)=C2C(NCC3=CC=CC=C3)=NC=N1 (tautomer)
    #  0 C1(C=CC=C2)=C2C(NCC3=CC=CC=C3)=NC=N1                +0    -6.025     0.496 *
    #  1 c1ccc(C[NH2+]c2ncnc3ccccc23)cc1                     +1    -2.920   0.00283
    #  2 c1ccc(CNc2[nH+]cnc3ccccc23)cc1                      +1    -2.741   0.00236
    #  3 c1ccc(CNc2nc[nH+]c3ccccc23)cc1                      +1    -2.964   0.00296
    #  4 c1ccc(C[N-]c2ncnc3ccccc23)cc1                       -1     7.657  4.45e-06
    #  5 c1ccc(C[NH2+]c2[nH+]cnc3ccccc23)cc1                 +2    19.674  5.55e-14
    #  6 c1ccc(C[NH2+]c2nc[nH+]c3ccccc23)cc1                 +2    21.270  1.12e-14
    #  7 c1ccc(CNc2ncnc3ccccc23)cc1                          +0    -6.025     0.496 *
    #  8 c1ccc(CNc2[nH+]c[nH+]c3ccccc23)cc1                  +2    11.912   1.3e-10
    #  9 c1ccc(C[N-]c2[nH+]cnc3ccccc23)cc1                   +0     7.562  6.23e-07
    # 10 c1ccc(C[N-]c2nc[nH+]c3ccccc23)cc1                   +0    10.144  4.71e-08
    # 11 c1ccc(C[NH2+]c2[nH+]c[nH+]c3ccccc23)cc1             +3    21.369   1.3e-15
    # 12 c1ccc(C[N-]c2[nH+]c[nH+]c3ccccc23)cc1               +1    12.133  8.21e-10
    