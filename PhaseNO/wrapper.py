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

        Args:
            net (BPMF.dataset.Network): Network object from BPMF, or a Pandas dataframe with
                a list of station names, latitude, and longitude of each station.
            graph_type (str | None): Either "knn" for a k-nearest neighbors graph or "radius"
                for distance based neighborhood graph. Use None for a fully connected graph (not recommended).
            k (int): Number if nearest neighbors if the graph is knn type. Not used for a radius graph.
            radius (float): neighborhood distance for a radius graph. Not used for a knn graph.
                Right now, the radius is in fraction of total domain. E.g., if the overall range of the
                seismic network is 2 degrees, a radius of 0.2 will be 0.4 degrees.

        Examples:
            First instantiate the class with the network file and graph type

            >>> ml_detector = PhaseNOPredictor(net, graph_type="knn", k=10)

            Then predict pick probabilties

            >>> probs = ml_detector(traces)
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

    def __call__(self, traces):
        """Generate predictions from seismic waveforms

        Args:
            traces (numpy.ndarray | torch.Tensor): (n stations x n channels x n samples) dimensional array/tensor
                of seismic data
        Returns:
            numpy.ndarray : An (n stations x 2 x nsamples)-dimensional array of pick probabilties.
        """
        # need to normalize?
        # X is a (num_station x 5 x nsamples) tensor
        # station index, [E trace, N trace, Z trace, x position, y position], samples

        # X_ should be (n_stations x 3 (ENZ) x n_samples array) of data
        # then need to add the x and y position axes to it

        if type(traces) != torch.Tensor:
            traces = torch.tensor(traces, dtype=torch.float)

        # strategy for dealing with data not 3000 samples in length
        # pad to make sure length is multiple of 3000, then
        # reshape into n batches of 3000
        traces = self._pad_and_reshape(traces)

        # normalize after breaking into segments to get better
        # local normalization
        traces = self._normalize(traces)

        # add coordinates to each station
        X = self._add_coords(traces)

        X = X.float().to(self.device)

        return self._predict(X)

    def _pad_and_reshape(self, data):
        # Reshapes data into batches of 3000-sample-long traces
        # pads with zeros to ensure total length is multiple of 3000

        # need extra padding if total samples in the following range
        if data.shape[-1] >= 3000 and data.shape[-1] < 6000:
            self.pad_length = 6000 - data.shape[-1] % 3000
        else:
            self.pad_length = 3000 - data.shape[-1] % 3000

        data = torch.nn.functional.pad(data, (0,self.pad_length), mode="constant", value=0)

        # 3000 sample traces, 1000 sample overlap
        data = data.unfold(-1, 3000, 2000)
        data = torch.movedim(data, (0,1,2,3), (1,2,0,3))

        return data

    def _normalize(self, traces):
        # normalize traces segment-wise
        eps = 1e-6
        traces_mean = torch.mean(traces,axis=-1,keepdims=True)
        traces_std  = torch.std(traces,axis=-1,keepdims=True)
        traces = (traces-traces_mean)/(traces_std+eps)
        # why divide by 10 - seems to get into better range
        traces /= 10

        return traces

    def _add_coords(self, traces):
        # stack coordinates of station into the trace array
        coords = np.stack([self.net.x, self.net.y]).T
        coords = torch.from_numpy(coords).float()

        # matches dims 0 and 2 to X_ dims:
        coords = coords.unsqueeze(-1).unsqueeze(0)
        coords = coords.repeat((traces.shape[0],1,1,traces.shape[-1]))
        X = torch.concat([traces, coords], dim=2)

        return X

    def _predict(self, X):
        # Currently, PhaseNO doesn't seem to predict on batches
        # loops over each segment individually
        # TODO: figure out how to make PhaseNO predict on batches, will speed up
        preds = torch.Tensor()

        for i, batch in enumerate(X):
            pred = torch.sigmoid(self.model.forward((batch, None, self.edge_index)))
            if i==0:
                preds = torch.concatenate([preds, pred], -1)
            else: # chop off first 1000 samples, they are the overlap
                preds = torch.concatenate([preds, pred[:,:,1000:]], -1)

        if self.pad_length == 0:
            return preds
        else:
            return preds[...,:-self.pad_length]