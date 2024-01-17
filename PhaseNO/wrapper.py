# wrapper for to use as a detector for BPMF

import warnings

import numpy as np
import torch
from PhaseNO import PhaseNO
from PhaseNO.utilities import (
    generate_edge_index,
    generate_knn_edge_index,
    generate_radius_edge_index
)


class PhaseNOPredictor:

    def __init__(self, net, graph_type=None, k=10, radius=1):
        """PhaseNO predictor for BPMF

        Takes a BPMF network object at initialization to create the graph.
        Can select the graph type

        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net
        self.edge_index, station_coords = self._make_graph(graph_type=graph_type, k=k, radius=radius)
        self.net["x"] = station_coords.T[0]
        self.net["y"] = station_coords.T[1]

        self.model = PhaseNO.load_from_checkpoint('../models/epoch=19-step=1140000.ckpt').to(self.device)

    def _make_graph(self, graph_type=None, k=10, radius=1):
        # make the graph and return edge index tensor

        if graph_type == "knn":
            # make knn graph
            return generate_knn_edge_index(self.net, k=k)
        if graph_type == "radius":
            return generate_radius_edge_index(self.net, radius=radius)
        else:
            warnings.warn("You are creating a fully connected graph, which can be very \
                          slow and memory intensive for a large number of stations. \
                          Consider a 'knn' or 'radius' graph_type instead.")
            return generate_edge_index(self.net)

    def __call__(self, X_):
        # need to normalize?
        # X is a (num_station x 5 x nsamples) tensor
        # station index, [E trace, N trace, Z trace, x position, y position], samples

        # X_ should be (n_stations x 3 (ENZ) x n_samples array) of data
        # then need to add the x and y position axes to it

        eps = 1e-6
        # normalize
        X_mean = torch.mean(X_,axis=-1,keepdims=True)
        X_std  = torch.std(X_,axis=-1,keepdims=True)
        X_ = (X_-X_mean)/(X_std+eps)
        # why divide by 10?
        X_ /= 10

        # add coordinates to each station
        Y = np.stack([self.net.x, self.net.y]).T
        Y = torch.from_numpy(Y).float()
        Y = Y.unsqueeze(-1).repeat((1,1,X_.shape[-1])) # matches dims 0 and 2 to X_ dims
        X = torch.concat([X_, Y], dim=1)

        if type(X) != torch.Tensor:
            X = torch.tensor(X, dtype=torch.float)
        X = X.float().to(self.device)
        return torch.sigmoid(self.model.forward((X,None,self.edge_index)))



