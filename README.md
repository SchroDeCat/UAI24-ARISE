# UAI24-ARISE

This repository contains the implementation of the Adaptive Region of Interest Search for Nash Equilibrium (ARISE) algorithm. The ARISE algorithm was proposed in the paper [No-Regret Learning of Nash Equilibrium for Black-Box Games via Gaussian Processes](https://arxiv.org/abs/2405.08318), which was published at the 40th Conference on Uncertainty in Artificial Intelligence (UAI 2024).

```
@InProceedings{han24no-regret,
  title = 	 {No-Regret Learning of Nash Equilibrium for Black-Box Games via Gaussian Processes},
  author =       {Han, Minbiao and Zhang, Fengxue and Chen, Yuxin},
  booktitle = 	 {Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence},
  year = 	 {2024},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  pdf = 	 {https://arxiv.org/pdf/2405.08318},
  url = 	 {https://arxiv.org/abs/2405.08318}
}
```

## Environment

The implementation has been tested on M1 Pro with 16GB RAM and macOS 14.2.1 (23C71). Using the following code to install the conda environment:

```shell
conda env create -f environment.yml
```

## Here are instructions for running the algorithm

```shell
# Generate visualization on pre-computed results 
python visualize_uai_rebuttal.py

# All algorithms (Hotelling)
python test_3_player.py --task=hotelling  --train_iter=10 --n_repeat=10 --opt_steps=200 --lr=1e-2 --retrain_interval=1 --interpolate --n_init=10

# All algorithms (BudgetAllocation)
python test_3_player.py --task=BudgetAllocation  --train_iter=10 --n_repeat=10 --opt_steps=200 --lr=1e-2 --retrain_interval=1 --interpolate --n_init=10
```
