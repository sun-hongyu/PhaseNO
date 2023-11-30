# PhaseNO
Phase Neural Operator for Multi-Station Phase Picking from Dynamic Seismic Networks.

![Method](https://github.com/sun-hongyu/PhaseNO/blob/master/phaseno.png)

## 1. Citation
```
Sun, H., Ross, Z.E., Zhu, W. and Azizzadenesheli, K., 2023. Phase Neural Operator for Multi-Station Picking of Seismic Arrivals. arXiv preprint arXiv:2305.03269.
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

1. phaseno_predict.ipynb
Use the pre-trained model to pick phases from one-hour continuous data of the 2019 Ridgecrest earthquake sequence.

2. phaseno_plot.ipynb
Plot the predicted probabilities and picks for all stations.


