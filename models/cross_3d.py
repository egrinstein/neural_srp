"""
	Pytorch models for DOA estimation.

	File name: acousticTrackingModels.py
	Author: David Diaz-Guerra
	Date creation: 05/2020
	Python Version: 3.8
	Pytorch Version: 1.4.0
"""

import torch
import torch.nn as nn

from models.layers import SphericPad, CausConv3d, CausConv1d


class Cross3D(nn.Module):
    """Proposed model with causal 3D convolutions and two branches with different pooling in each axis SRP-PHAT map axis."""

    def __init__(
        self,
        res_the,
        res_phi,
        in_deep=1,
        in_krnl_sz=(5, 5, 5),
        in_nb_ch=32,
        pool_sz=(1, 1, 1),
        cr_deep=4,
        crThe_krnl_sz=(5, 3, 3),
        crPhi_krnl_sz=(5, 3, 3),
        cr_nb_ch=32,
        crThe_pool_sz=(1, 1, 2),
        crPhi_pool_sz=(1, 2, 1),
        out_conv_len=5,
        out_conv_dilation=2,
        out_nbh=128,
    ):
        """
        res_the: elevation resolution of the input maps
        res_phi: azimuth resolution of the input maps
        in_deep: Number of convolutional layers at the input [default: 1]
        in_krnl_sz: Kernel size of the convolutional layers at the input [default: (5,5,5)]
        in_nb_ch: Number of channels of the convolutional layers at the input [default: 32]
        pool_sz: Kernel size and stride of the max pooling layer after the initial 3D CNNs [default: (1,1,1)]
        cr_deep: Number of convolutional layers to apply in cross branches [default: 4]
        crThe_krnl_sz: Kernel size of the convolutional layers at the theta branch of the cross [default: (5,3,3)]
        crPhi_krnl_sz: Kernel size of the convolutional layers at the phi branch of the cross [default: (5,3,3)]
        cr_nb_ch: Number of channels of the convolutional layers at the cross branches [default: 32]
        crThe_pool_sz: Kernel size and stride of the max pooling layer between each convolutional layer of the theta branch [default: (1,1,2)]
        crPhi_pool_sz: Kernel size and stride of the max pooling layer between each convolutional layer of the theta branch [default: (1,2,1)]
        out_conv_len: Kernel size of the two 1D convolutional layers at end of the network [default: 5]
        out_conv_dilation: Dilation of the two 1D convolutional layers at end of the network [default: 2]
        out_nbh: Number of channels of the first of the two 1D convolutional layers at end of the network [default: 128]
        """

        super().__init__()

        self.res_the = res_the
        self.res_phi = res_phi
        self.in_deep = in_deep
        self.cr_deep = cr_deep
        self.in_nb_ch = in_nb_ch
        self.cr_nb_ch = cr_nb_ch

        self.crThe_resThe = res_the // pool_sz[1] // crThe_pool_sz[1] ** cr_deep
        self.crThe_resPhi = res_phi // pool_sz[2] // crThe_pool_sz[2] ** cr_deep
        self.crPhi_resThe = res_the // pool_sz[1] // crPhi_pool_sz[1] ** cr_deep
        self.crPhi_resPhi = res_phi // pool_sz[2] // crPhi_pool_sz[2] ** cr_deep
        self.crThe_nb_outAct = self.crThe_resThe * self.crThe_resPhi * cr_nb_ch
        self.crPhi_nb_outAct = self.crPhi_resThe * self.crPhi_resPhi * cr_nb_ch

        self.in_sphPad = SphericPad(
            (in_krnl_sz[2] // 2,) * 2 + (in_krnl_sz[1] // 2,) * 2
        )
        self.in_conv = nn.ModuleList(
            [CausConv3d(3, in_nb_ch, in_krnl_sz)] if in_deep > 0 else []
        )
        self.in_conv += nn.ModuleList(
            [
                CausConv3d(in_nb_ch, in_nb_ch, in_krnl_sz)
                for i in range(in_deep - 1)
            ]
        )
        self.in_prelu = nn.ModuleList([nn.PReLU(in_nb_ch) for i in range(in_deep)])

        self.pool = nn.MaxPool3d(pool_sz)

        self.crThe_sphPad = SphericPad(
            (crThe_krnl_sz[2] // 2,) * 2 + (crThe_krnl_sz[1] // 2,) * 2
        )
        self.crThe_conv = nn.ModuleList(
            [CausConv3d(in_nb_ch, cr_nb_ch, crThe_krnl_sz)]
            if cr_deep > 0
            else []
        )
        self.crThe_conv += nn.ModuleList(
            [
                CausConv3d(cr_nb_ch, cr_nb_ch, crThe_krnl_sz)
                for i in range(cr_deep - 1)
            ]
        )
        self.crThe_prelu = nn.ModuleList([nn.PReLU(cr_nb_ch) for i in range(cr_deep)])
        self.crThe_pool = nn.MaxPool3d(crThe_pool_sz)

        self.crPhi_sphPad = SphericPad(
            (crPhi_krnl_sz[2] // 2,) * 2 + (crPhi_krnl_sz[1] // 2,) * 2
        )
        self.crPhi_conv = nn.ModuleList(
            [CausConv3d(in_nb_ch, cr_nb_ch, crPhi_krnl_sz)]
            if cr_deep > 0
            else []
        )
        self.crPhi_conv += nn.ModuleList(
            [
                CausConv3d(cr_nb_ch, cr_nb_ch, crPhi_krnl_sz)
                for i in range(cr_deep - 1)
            ]
        )
        self.crPhi_prelu = nn.ModuleList([nn.PReLU(cr_nb_ch) for i in range(cr_deep)])
        self.crPhi_pool = nn.MaxPool3d(crPhi_pool_sz)

        self.out_conv1 = CausConv1d(
            self.crThe_nb_outAct + self.crPhi_nb_outAct,
            out_nbh,
            out_conv_len,
            dilation=out_conv_dilation,
        )
        self.out_prelu = nn.PReLU()
        self.out_conv2 = CausConv1d(
            out_nbh, 3, out_conv_len, dilation=out_conv_dilation
        )

    def forward(self, x):
        x = x["signal"]

        for i in range(self.in_deep):
            x = self.in_prelu[i](self.in_conv[i](self.in_sphPad(x)))
        x = self.pool(x)

        xThe = x
        xPhi = x
        for i in range(self.cr_deep):
            xThe = self.crThe_prelu[i](
                self.crThe_pool(self.crThe_conv[i](self.crThe_sphPad(xThe)))
            )
            xPhi = self.crPhi_prelu[i](
                self.crPhi_pool(self.crPhi_conv[i](self.crPhi_sphPad(xPhi)))
            )

        xThe = (
            xThe.transpose(1, 2)
            .contiguous()
            .view(-1, x.shape[-3], self.crThe_nb_outAct)
        )
        xPhi = (
            xPhi.transpose(1, 2)
            .contiguous()
            .view(-1, x.shape[-3], self.crPhi_nb_outAct)
        )

        x = torch.cat((xThe, xPhi), dim=2)
        x = x.transpose(1, 2)

        x = self.out_prelu(self.out_conv1(x))
        x = torch.tanh((self.out_conv2(x)))
        x = x.transpose(1, 2)

        return {
            "doa_cart": x
		}
