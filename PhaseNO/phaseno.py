import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import random; random.seed(10)

import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch_geometric.nn import MessagePassing

import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Feedforward(torch.nn.Module):
        def __init__(self, channel_size, hidden_size):
            super(Feedforward, self).__init__()
            self.channel = channel_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.channel*2+4, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.hidden_size, self.channel)

        def forward(self, x): #[E, in_channels, NT]
            x = x.permute(0,2,1)
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            return x.permute(0,2,1)

class Feedforward_aggre(torch.nn.Module):
        def __init__(self, channel_size, hidden_size):
            super(Feedforward_aggre, self).__init__()
            self.channel = channel_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.channel*2, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.hidden_size, self.channel)

        def forward(self, x): #[E, in_channels, NT]
            x = x.permute(0,2,1)
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            return x.permute(0,2,1)

class GNO(MessagePassing):
    def __init__(self, in_channels):
        super().__init__(aggr='mean',node_dim=-3)
        activation= nn.GELU()
        dropout=0.1
        hidden = in_channels*2*2
        self.edge  = Feedforward(in_channels,hidden)
        self.aggre = Feedforward_aggre(in_channels,hidden)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index[:2,:].to(torch.int64), x=x, xloc=edge_index[2:,:])

    def message(self, x_i, x_j, xloc):

        #x_i.shape: torch.Size([E, channels, NT])
        nedge = xloc.shape[1]
        loc_i = torch.zeros((nedge,2,x_i.shape[-1]),device=x_i.device)
        loc_j = torch.zeros((nedge,2,x_j.shape[-1]),device=x_j.device)
        for e in np.arange(nedge):
            loc_i[e,0,:] = xloc[0,e]
            loc_i[e,1,:] = xloc[1,e]
            loc_j[e,0,:] = xloc[2,e]
            loc_j[e,1,:] = xloc[3,e]

        # concatenate x_i and x_j along channel dimension
        tmp = torch.cat([x_i, loc_i, x_j, loc_j], dim=1)  # tmp has shape [E, 2 * in_channels+4, NT]
        ans = self.edge(tmp)
        return ans

    def update(self, aggr_out, x):
        #aggr_out.shape: torch.Size([5, 16, 6100])
        #x.shape: torch.Size([5, 16, 6100])
        tmp = torch.cat([aggr_out, x], dim=1)  # tmp has shape [E, 2 * in_channels, NT]
        ans = self.aggre(tmp)
        return ans

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, dim1, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.dim1 = dim1
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand((in_channels, out_channels, self.modes1), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, modes on t), (in_channel, out_channel, modes on t) -> (batch, out_channel, modes on t)

        #print('einsum input', input.shape)
        #print('einsum weights', weights.shape)

        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1=None):

        if dim1 is not None:
            self.dim1 = dim1

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, norm = 'forward')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  self.dim1//2 + 1 , dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=self.dim1 , norm = 'forward')
        return x

class pointwise_op_1D(nn.Module):
    """
    All variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1):
        super(pointwise_op_1D,self).__init__()
        self.conv = nn.Conv1d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)

    def forward(self,x, dim1 = None):
        if dim1 is None:
            dim1 = self.dim1
        x_out = self.conv(x)

        x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True)#, antialias= True)
        return x_out

class FNO1D(nn.Module):
    """
    Normalize = if true performs InstanceNorm1d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv1d_Uno class.
    """
    def __init__(self, in_codim, out_codim, modes1, dim1, Normalize = True, Non_Lin = True):
        super(FNO1D,self).__init__()
        self.conv = SpectralConv1d(in_codim, out_codim, int(dim1), int(modes1))
        self.w = pointwise_op_1D(in_codim, out_codim, int(dim1))
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm1d(int(out_codim),affine=True)

    def forward(self,x):#, dim1 = None):
        """
        input shape = (batch, in_codim, input_dim1)
        output shape = (batch, out_codim, dim1)
        """
        x1_out = self.conv(x)#,dim1)
        x2_out = self.w(x)#,dim1)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class PhaseNO(pl.LightningModule):

    def __init__(self, modes = 24, width = 48):
        super(PhaseNO,self).__init__()

        self.modes1 = modes
        self.width  = width
        self.padding = 50 # pad the domain if input is non-periodic

        self.criterion =  torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]))
        self.loss_weights = [0.5,0.5]

        self.fc0 = nn.Linear(6, self.width)

        self.fno0 = FNO1D(self.width, self.width, self.modes1, 3000+self.padding*2)
        self.gno0 = GNO(self.width)

        self.fno1 = FNO1D(self.width, self.width*2, self.modes1/2, 750)
        self.gno1 = GNO(self.width*2)

        self.fno2 = FNO1D(self.width*2, self.width*4, self.modes1/3, 200)
        # self.gno2 = GNO(self.width*4)

        # self.fno3 = FNO1D(self.width*4, self.width*4, self.modes1/3, 200)
        self.gno3 = GNO(self.width*4)

        self.fno4 = FNO1D(self.width*8, self.width*2, self.modes1/3, 750)
        self.gno4 = GNO(self.width*2)

        self.fno5 = FNO1D(self.width*4, self.width*1, self.modes1/2, 3000+self.padding*2)
        self.gno5 = GNO(self.width*1)

        self.fno6 = FNO1D(self.width*2, self.width, self.modes1, 3000+self.padding*2)

        self.fno7 = FNO1D(self.width, self.width, self.modes1, 3000+self.padding*2, Non_Lin = False)

        self.fc1 = nn.Linear(self.width, self.width*2)
        self.fc2 = nn.Linear(self.width*2, 2)

    def forward(self, data):

        """
        By default, the predict_step() method runs the forward() method.
        In order to customize this behaviour, simply override the predict_step() method.
        trainer = Trainer()
        predictions = trainer.predict(model, dataloaders=test_dataloader)
        """
        x, edge_index = data[0], data[2]

        #x shape: [nstation, nchannel, nt]
        grid = self.get_grid(x.shape,x.device)

        x = torch.cat((x, grid), dim=1)
        x = F.pad(x, [self.padding,self.padding], mode ='reflect', value=0) # pad the domain if input is non-periodic

        x = np.squeeze(x) # nstation, nchannel, nt
        edge_index=np.squeeze(edge_index) # 2, nedge

        x = x.permute(0, 2, 1) # nstation, nt, nchannel
        x = self.fc0(x)
        x = x.permute(0, 2, 1) # nstation, nchannel,nt

        x0 = self.fno0(x)
        x = self.gno0(x0,edge_index)
        x1 = self.fno1(x)
        x = self.gno1(x1,edge_index)
        x2 = self.fno2(x)

        # x = self.gno2(x2,edge_index)
        # x = self.fno3(x)
        x = self.gno3(x2,edge_index)
        x = torch.cat([x2, x], dim=1)

        x = self.fno4(x)
        x = self.gno4(x,edge_index)
        x = torch.cat([x1, x], dim=1)

        x = self.fno5(x)
        x = self.gno5(x,edge_index)
        x = torch.cat([x0, x], dim=1)

        x = self.fno6(x)
        x = self.fno7(x)

        x = x[...,self.padding:-self.padding]

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) #nstation, nt, nchannel

        return x.permute(0, 2, 1)

    def get_grid(self, shape, device):

        nstation, size_t = shape[-3], shape[-1]   # nstation, nchannel, nt
        gridx = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_t).repeat([nstation, 1, 1])
        return gridx.to(device)

    def training_step(self, batch, batch_idx):

        y = batch[1]
        y = np.squeeze(y)

        y_hat = self.forward(batch)

        y_hatP = y_hat[:,0,:].reshape(-1,1)
        yP = y[:,0,:].reshape(-1,1)
        lossP = self.criterion(y_hatP, yP)* self.loss_weights[0]

        y_hatS = y_hat[:,1,:].reshape(-1,1)
        yS = y[:,1,:].reshape(-1,1)
        lossS = self.criterion(y_hatS, yS)* self.loss_weights[1]

        loss = lossP+lossS
        # Update loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_lossP", lossP, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_lossS", lossS, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, #must be this name
                "train_lossP": lossP,
                "train_lossS": lossS
                }

    def validation_step(self, batch, batch_idx):

        y = batch[1]
        y = np.squeeze(y)

        y_hat = self.forward(batch)

        y_hatP = y_hat[:,0,:].reshape(-1,1)
        yP = y[:,0,:].reshape(-1,1)
        lossP = self.criterion(y_hatP, yP)* self.loss_weights[0]

        y_hatS = y_hat[:,1,:].reshape(-1,1)
        yS = y[:,1,:].reshape(-1,1)
        lossS = self.criterion(y_hatS, yS)* self.loss_weights[1]

        loss = lossP+lossS

        # Update loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_lossP", lossP, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_lossS", lossS, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss,
                "val_lossP": lossP,
                "val_lossS": lossS
        }

    def configure_optimizers(self):
        # build optimizer and schedule learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss",
                    "interval": "epoch",
                    "frequency": 100,
                       },
        }

