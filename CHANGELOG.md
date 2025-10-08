# ChangeLog

## Oct 8, 2025

- forked from https://github.com/dptech-corp/Uni-pKa (Apache-2.0 license)
- published by W. Luo, et al., Bridging machine learning and thermodynamics for accurate pK a prediction. [JACS Au 4, 3451–3465 (2024)](https://pubs.acs.org/doi/10.1021/jacsau.4c00271).
- packaged with pyproject
- pretrained model weight file (`model/t_dwar_v_novartis_a_b.pt`; 545 MB) was downloaded from [bohrium](https://www.bohrium.com/notebooks/38543442597). It is handled by the GitHub LFS because of its size exceeding the limit of 100 MB. You can use `git clone` to install the codes and download the pretrained model weight from the github repository:
    ```sh
    git clone https://github.com/sunghunbae/unipka.git
    cd unipka
    pip install .
    ```