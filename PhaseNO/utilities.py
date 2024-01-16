import numpy as np
import torch
from torch_geometric.nn import knn_graph


def normalize_positions(stations):
    """Converts lat/lon station positions to relative locations
    in the domain ((0,0), (1,1))
    """

    # lat_max = stations.latitude.max()
    lat_min = stations.latitude.min()
    # lon_max = stations.longitude.max()
    lon_min = stations.longitude.min()

    x_min, y_min = lon_min - 1, lat_min - 1 # gives a 1 degree buffer

    station_coords = []

    nstations = len(stations.station.values)

    for i, sta in enumerate(stations.station.values):
        station_0_1_ = [(stations.loc[stations['station'] == sta].iloc[0]['longitude']- x_min)/2,
                        (stations.loc[stations['station'] == sta].iloc[0]['latitude']- y_min)/2]
        station_coords.append(station_0_1_)
    station_coords = np.array(station_coords)

    return station_coords


def generate_edge_index(stations):
    # generate edge_index
    # stations is a dataframe listing stations with lat and lon (at least)
    # center is the center coordinate of
    # See phaseno_predict.ipynb for current structure
    # TODO: change input stations data structure to BPMF

    nstations = len(stations.station.values)
    station_coords = normalize_positions(stations)

    row_a=[]
    row_b=[]
    row_ix=[]
    row_iy=[]
    row_jx=[]
    row_jy=[]

    for i in range(nstations):
        for j in range(nstations):
            row_a.append(i)
            row_b.append(j)
            row_ix.append(station_coords[i,0])
            row_iy.append(station_coords[i,1])
            row_jx.append(station_coords[j,0])
            row_jy.append(station_coords[j,1])

    edge_index=[row_a,row_b,row_ix,row_iy,row_jx,row_jy]
    edge_index = torch.from_numpy(np.array(edge_index)).float()

    return edge_index, station_coords

def generate_knn_edge_index(stations, k=4):
    """Creates an edge index of k-nearest neighbors (or ball tree)
    For a NN graph network. Reduces computational time wrt fully connected.

    Returns and edge_index as a Torch.tensor
    """

    station_coords = normalize_positions(stations)
    coords = torch.from_numpy(station_coords).float()
    edge_index = knn_graph(coords, k=k)

    # Get coordinates of each vertex in the edges and build the edge_index
    # data structure needed for GNO
    i_coords = torch.from_numpy(station_coords[edge_index[0]])
    j_coords = torch.from_numpy(station_coords[edge_index[1]])
    edge_index = torch.vstack([edge_index, i_coords.T, j_coords.T])

    return edge_index, station_coords