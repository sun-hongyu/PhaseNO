# PhaseNO
Phase Neural Operator for Multi-Station Phase Picking from Dynamic Seismic Networks.

![Method](https://github.com/sun-hongyu/PhaseNO/blob/master/phaseno.png)

## 1. Citation
```
Sun, H., Ross, Z.E., Zhu, W. and Azizzadenesheli, K., 2023. Next-Generation Seismic Monitoring with Neural Operators. arXiv preprint arXiv:2305.03269.
```

## 2. Installation

Create an environment with conda for PhaseNO
```
conda env create -f env.yml
conda activate phaseno
```

## 3. Pre-trained model
Located in directory: models/*.ckpt

## 4. Example 
Located in directory: example

Use the pre-trained model to pick phases from one-hour continuous data of the 2019 Ridgecrest earthquake sequence.

There are two scripts that can be used in the following order to reproduce the results in this example:

1. phaseno_predict.ipynb
2. phaseno_plot.ipynb
