#
# The SELDnet architecture
# Credits to https://github.com/sharathadavanne/doa-net
#

import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):

        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.relu_(self.bn(self.conv(x)))
        return x


class DoaNet(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        """
        Initialize the CRNN model.
        :param in_feat_shape: Shape of the input features.
        :param out_shape: Shape of the output.
        :param params: Dictionary of parameters.
        """

        super().__init__()
        self.use_activity_out = params['use_activity_output']
        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    nn.Dropout2d(p=params['dropout_rate'])
                )

        self.in_gru_size = int(params['nb_cnn2d_filt'] * (in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.fnn_list = nn.ModuleList()
        for fc_cnt in range(params['nb_fnn_layers']):
            self.fnn_list.append(
                nn.Linear(params['fnn_doa_size'] if fc_cnt else params['rnn_size'] , params['fnn_doa_size'], bias=True)
            )
        self.fnn_list.append(
            nn.Linear(params['fnn_doa_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )

        # Branch for activity detection
        self.n_max_sources = out_shape[-1]//3
        if self.use_activity_out:
            self.fnn_act_list = nn.ModuleList()
            for fc_cnt in range(params['nb_fnn_act_layers']):
                self.fnn_act_list.append(
                    nn.Linear(params['fnn_act_size'] if fc_cnt else params['rnn_size'] , params['fnn_act_size'], bias=True)
                )
            self.fnn_act_list.append(
                nn.Linear(params['fnn_act_size'] if params['nb_fnn_act_layers'] else params['rnn_size'], self.n_max_sources, bias=True)
            )


    def forward(self, x):
        x = x['signal'][..., 0] # ignore 'metadata', unlike Neural-SRP
        # Also, ignore channel dimension

        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        '''(batch_size, time_steps, feature_maps)'''
   
        x_rnn = x
        for fnn_cnt in range(len(self.fnn_list)-1):
            x = torch.relu_(self.fnn_list[fnn_cnt](x))
        doa = torch.tanh(self.fnn_list[-1](x))
        '''(batch_size, time_steps, label_dim)'''

        # Reshape the output to be (batch_size, time_steps, n_sources, 3),
        doa = doa.view(
            doa.shape[0], doa.shape[1], 3, self.n_max_sources
        ).transpose(-1, -2)
        
        out = {
            "doa_cart": doa
        }

        if self.use_activity_out:
            for fnn_cnt in range(len(self.fnn_act_list)-1):
                x_rnn = torch.relu_(self.fnn_act_list[fnn_cnt](x_rnn))
            activity = self.fnn_act_list[-1](x_rnn)

            out["activity"] = activity
        
        return out
