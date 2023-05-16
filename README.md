# PhaseNO
Phase Neural Operator for Multi-Station Phase Picking from Dynamic Seismic Networks.

## 1. Citation:
```
Sun, H., Ross, Z.E., Zhu, W. and Azizzadenesheli, K., 2023. Next-Generation Seismic Monitoring with Neural Operators. arXiv preprint arXiv:2305.03269.
```

## 2. Pre-trained model
Located in directory: models/*.ckpt

## 3. Example 
Located in directory: example

Use the pre-trained model to pick phases from one-hour continous data from the 2019 Ridgecrest earthquake sequence.

There are three scripts that should be used in the following order to reproduce the results in this example

1. phaseno_predict.ipynb
2. phaseno_postprocess.ipynb
3. phaseno_plot.ipynb



