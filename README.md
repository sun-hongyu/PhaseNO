[![DOI](https://zenodo.org/badge/641315064.svg)](https://zenodo.org/doi/10.5281/zenodo.10224300)

# PhaseNO 
Phase Neural Operator for Multi-Station Phase Picking from Dynamic Seismic Networks.

![Method](https://github.com/sun-hongyu/PhaseNO/blob/master/phaseno.png)


## Update Log

**Version 1.0.1 (March 31, 2025):**
- This version introduces a physical distance constraint (`dis_range`) between stations to limit information exchange to only nearby stations.
- This update significantly reduces computational cost without compromising picking performance. It better reflects realistic spatial relationships, as earthquake signals recorded at one station are most likely to appear at surrounding stations within a certain distance.
- This resolves a major limitation in PhaseNO v1.0.0, where computational cost (in terms of memory usage and speed) scaled quadratically with the number of stations. In the current version, such quadratic scaling only occurs if the user sets the distance threshold `dis_range` larger than the maximum distance between any pair of stations, effectively enabling full communication among all nodes. However, such fully connected communication is generally unnecessary.
- Users should adjust the new paremeter (`dis_range`) according to their seismic network configuration. The default is set to 30 km, meaning each station will only communicate with stations within that distance. Reducing this value can significantly lower computational demands.
- With a reasonable choice of `dis_range`, users can now include all stations in a single run, without needing to randomly select a subset from a large seismic network. 

## Citation
```
Sun, H., Ross, Z.E., Zhu, W. and Azizzadenesheli, K., 2023. Phase Neural Operator for Multi-Station Picking of Seismic Arrivals. arXiv preprint arXiv:2305.03269.
```

## Installation

Create an environment with conda for PhaseNO
```
conda env create -f env.yml
conda activate phaseno
```

## Pre-trained model
Located in directory: models/*.ckpt

## Example 
Located in directory: example

- phaseno_predict.ipynb
  
  Use the pre-trained model to pick phases from one-hour continuous data of the 2019 Ridgecrest earthquake sequence.

- phaseno_plot.ipynb
  
  Plot the predicted probabilities and picks for all stations.


