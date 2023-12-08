# https://github.com/sharathadavanne/hungarian-net

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return (
            self._get_name()
            + "(in_channels={}, out_channels={}, key_channels={})".format(
                self.conv_Q.in_channels,
                self.conv_V.out_channels,
                self.conv_K.out_channels,
            )
        )


class HNetGRU(nn.Module):
    def __init__(self, max_len=4):
        super().__init__()
        hidden_size = 128
        self.nb_gru_layers = 1
        self.gru = nn.GRU(max_len, hidden_size, self.nb_gru_layers, batch_first=True)
        self.attn = AttentionLayer(hidden_size, hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, max_len)

    def forward(self, query):
        # query - batch x seq x feature

        out, _ = self.gru(query)
        # out - batch x seq x hidden

        out = out.permute((0, 2, 1))
        # out - batch x hidden x seq

        out = self.attn.forward(out)
        # out - batch x hidden x seq

        out = out.permute((0, 2, 1))
        out = torch.tanh(out)
        # out - batch x seq x hidden

        out = self.fc1(out)
        # out - batch x seq x feature

        out1 = out.view(out.shape[0], -1)
        # out1 - batch x (seq x feature)

        out2, _ = torch.max(out, dim=-1)

        out3, _ = torch.max(out, dim=-2)

        # out2 - batch x seq x 1
        return out1.squeeze(), out2.squeeze(), out3.squeeze()

