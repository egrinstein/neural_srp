import numpy as np
import torch
import torch.nn as nn

# from torchinfo import summary

from datasets.mic_pos_utils import get_all_pairs, prepare_mic_pos
from models.signal_processing import GCC, Window
from models.layers import Mlp
from models.mic_selection import select_pairs


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                        pool_size=None, dropout_rate=0.0):

        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

        if pool_size is not None:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))

        if hasattr(self, "pool"):
            x = self.pool(x)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return x


class ConditionalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_metadata=0, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 pool_size=None, dropout_rate=0.0):

        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

        if pool_size is not None:
            self.pool = nn.MaxPool2d(kernel_size=pool_size)
        
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

        if n_metadata > 0:
            self.metadata_proj = nn.Linear(
                n_metadata, out_channels
            )

    def forward(self, x, metadata=None):

        x = self.conv(x)

        # Add metadata as bias
        if metadata is not None and hasattr(self, "metadata_proj"):
            metadata = self.metadata_proj(metadata)
            metadata = metadata.unsqueeze(-1).unsqueeze(-1)
            x = x + metadata

        x = self.activation(self.bn(x))

        if hasattr(self, "pool"):
            x = self.pool(x)

        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x


class NeuralSrp(nn.Module):
    def __init__(self, n_gcc_bins, params, n_max_sources=2, n_max_dataset_sources=2):
        """
        Initialize the CRNN model.
        :param n_gcc_bins: Number of the input gcc_bins.
        :param params: Dictionary of parameters.
        :param n_max_sources: Maximum number of simulataneous sources.
        """

        super().__init__()

        self.n_max_sources = n_max_sources
        self.n_max_dataset_sources = n_max_dataset_sources

        self.use_activity_out = params["use_activity_output"]
        self.bidirectional_rnn = params["bidirectional_rnn"]
        self.pair_agg_mode = params["pair_agg_mode"]
        self.use_batch_norm_hidden = params["use_batch_norm_hidden"]
        
        self.conv_block_list = nn.ModuleList()
        
        self.metadata_type = params["metadata_type"]
        self.metadata_fusion_mode = params["metadata_fusion_mode"]
        assert self.metadata_type in [
            "mic_positions",
            "mic_diff_vector",
            "norm_mic_diff_vector",
            "idx"
        ]
        assert self.metadata_fusion_mode in [
            "early_conv_bias",
            "late_concat",
        ]

        self.conv_agg_mode = params["conv_agg_mode"]

        if params["input_feature"] == "gcc":
            self.n_input_channels = 1 # Correlation values
                                      # Also tried 2 (respective bins)
        elif params["input_feature"] == "pairwise_mel_phase":
            self.n_input_channels = 2 # 1 for each mic

        if self.metadata_type == "idx":
            self.n_metadata = 1
        elif self.metadata_type == "mic_positions":
            self.n_metadata = 6 # 3 for each mic
        elif self.metadata_type == "norm_mic_diff_vector":
            self.n_metadata = 4 # (mic1 - mic2)/||mic1 - mic2||, ||mic1 - mic2||
        elif self.metadata_type == "mic_diff_vector":
            self.n_metadata = 3 # mic1 - mic2

        # Input batch normalization
        self.use_batch_norm_input = params["use_batch_norm_input"]
        if self.use_batch_norm_input:
            self.input_bn = nn.BatchNorm2d(self.n_input_channels)
            self.metadata_bn = nn.BatchNorm1d(1)

        # Convolutional blocks
        n_conv_input = self.n_input_channels

        n_conv_layers = len(params["f_pool_size"])
        for conv_cnt in range(n_conv_layers):
            self.conv_block_list.append(
                ConditionalConvBlock(
                    in_channels=params["nb_cnn2d_filt"] if conv_cnt else n_conv_input,
                    out_channels=params["nb_cnn2d_filt"],
                    n_metadata=self.n_metadata if self.metadata_fusion_mode == "early_conv_bias" else 0,
                    pool_size=(params["t_pool_size"][conv_cnt], params["f_pool_size"][conv_cnt]),
                    dropout_rate=params["dropout_rate"]
                )
            )
        
        self.n_rnn_features = params["rnn_size"]

        if self.conv_agg_mode == "flatten":
            self.in_gru_size = int(
                params["nb_cnn2d_filt"] * (n_gcc_bins / np.prod(params["f_pool_size"])))
        elif self.conv_agg_mode in ["sum", "mean", "prod", "max"]:
            self.in_gru_size = params["nb_cnn2d_filt"]
        
        self.gru = nn.GRU(input_size=self.in_gru_size, hidden_size=self.n_rnn_features,
                                num_layers=params["nb_rnn_layers"], batch_first=True,
                                dropout=params["dropout_rate"],
                                bidirectional=params["bidirectional_rnn"])

        self.n_output = n_max_sources*3 # 3 coordinates (x, y, z) for each source
        
        self.n_pairwise_input_features = self.n_rnn_features
        self.n_pairwise_output_features = params["fnn_pairwise_size"]
        if self.metadata_fusion_mode == "late_concat":
            self.n_pairwise_input_features += self.n_metadata
     
        if params["nb_pairwise_fnn_layers"] > 0:
            self.fnn_pairwise = Mlp(
                self.n_pairwise_input_features, self.n_pairwise_output_features,
                self.n_pairwise_output_features, params["nb_pairwise_fnn_layers"],
                activation="prelu",
                output_activation="prelu",
                batch_norm=params["use_batch_norm_hidden"])
        else:
            self.fnn_pairwise = nn.Identity()
            self.n_pairwise_output_features = self.n_pairwise_input_features
            
        if self.use_batch_norm_hidden:
            self.pairwise_bn = nn.BatchNorm1d(self.n_pairwise_output_features)

        self.fnn_doa = Mlp(
            self.n_pairwise_output_features, self.n_output,
            params["fnn_doa_size"], params["nb_fnn_layers"],
            activation="prelu",
            output_activation=nn.Tanh(),
            batch_norm=params["use_batch_norm_hidden"])

        # Branch for activity detection
        if self.use_activity_out:
            self.fnn_activity = Mlp(
                self.n_pairwise_output_features, n_max_sources,
                params["fnn_act_size"], params["nb_fnn_act_layers"],
                activation="prelu",
                output_activation=None,
                batch_norm=params["use_batch_norm_hidden"])

        # summary(self)

    def _pairwise_forward(self, x, metadata=None):
        """input:
        
        x: (batch_size, time_steps, n_features, n_channels)
        metadata: (batch_size, n_metadata)  
        """

        # Push channels to first axis

        x = x.moveaxis(3, 1)
        if self.use_batch_norm_input:
            x = self.input_bn(x)
            metadata = self.metadata_bn(metadata.unsqueeze(1))[:, 0]

        # 1. Apply convolutional blocks
        for conv_block in self.conv_block_list:
            x = conv_block(x, metadata=metadata)
        """(batch_size, feature_maps, time_steps, n_gcc_bins)"""

        x = x.transpose(1, 2).contiguous()

        if self.conv_agg_mode == "flatten":
            x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        elif self.conv_agg_mode == "sum":
            x = x.sum(dim=-1)
        elif self.conv_agg_mode == "mean":
            x = x.mean(dim=-1)
        elif self.conv_agg_mode == "prod":
            x = x.prod(dim=-1)
        elif self.conv_agg_mode == "max":
            x = x.max(dim=-1)[0]

        """ (batch_size, time_steps, feature_maps):"""

        # 2. Apply GRU block
        (x, _) = self.gru(x)
        x = torch.tanh(x)

        if self.bidirectional_rnn:
            x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        """(batch_size, time_steps, feature_maps)"""

        # 3. Concatenate microphone metadata, if any
        if metadata is not None and self.metadata_fusion_mode == "late_concat":
            metadata = torch.ones(
                (x.shape[0], x.shape[1], self.n_metadata)
            ).to(x.device)*metadata.unsqueeze(1)
            x = torch.cat((x, metadata), dim=2)

        # 4. Apply fully connected layers
        x = self.fnn_pairwise(x)
        if self.use_batch_norm_hidden:
            x = self.pairwise_bn(x.transpose(1, 2)).transpose(1, 2)

        return x

    def forward(self, x):
        """Forward pass of the network

        Args:
            x (dict): dictionary containing the "signal" and "mic_pos"
                keys. The "signal" key contains the input signal, with
                shape (batch_size, n_mic_pairs, n_samples, n_gcc_bins, n_channels).
                The "mic_pos" key contains the metadata, with shape
                (batch_size, n_mic_pairs, n_metadata).
        Returns:
            torch.Tensor: output of the network
        """
        metadata = x["mic_pos"]
        x = x["signal"]

        batch_size, n_mic_pairs, n_frames, n_features, n_channels = x.shape
        # 1. Apply pairwise CRNN and sum features from all pairs

        # Flatten batch and mic pair dimensions
        x = x.reshape(-1, n_frames, n_features, n_channels)
        metadata = metadata.reshape(-1, self.n_metadata)

        x = self._pairwise_forward(
            x,
            metadata=metadata
        )
        x = x.reshape(
            batch_size, n_mic_pairs, -1, self.n_pairwise_output_features
        )

        if self.pair_agg_mode == "sum":
            x = x.sum(dim=1)
        if self.pair_agg_mode == "mean":
            x = x.mean(dim=1)
        elif self.pair_agg_mode == "prod":
            x = x.prod(dim=1)

        x_0 = x
        # 2. Apply DOA fully connected branch
        doa = self.fnn_doa(x)
        # (batch_size, time_steps, label_dim)
        # Reshape the output to be (batch_size, time_steps, n_sources, 3),
        doa = doa.view(
            doa.shape[0], doa.shape[1], 3, self.n_max_sources
        ).transpose(-1, -2)
        # only select self.n_max_dataset_sources sources
        doa = doa[:, :, :self.n_max_dataset_sources]
        if doa.shape[2] == 1:
            # Squeeze the source dimension if there is only one source
            doa = doa[:, :, 0]

        out = {
            "doa_cart": doa
        }

        # 3. Apply activity fully connected branch, if required
        if self.use_activity_out:
            activity = self.fnn_activity(x_0)

            # only select self.n_max_dataset_sources sources
            activity = activity[:, :, :self.n_max_dataset_sources]
            # if activity.shape[2] == 1:
            #     # Squeeze the source dimension if there is only one source
            #     activity = activity[:, :, 0]
            
            out["activity"] = activity

        return out


class NeuralSrpFeatureExtractor(torch.nn.Module):
    def __init__(self, params): 
        super().__init__()
        
        self.mic_pair_sampling_mode = params["mic_pair_sampling_mode"]
        self.n_mic_pairs = params["n_mic_pairs"]
        self.metadata_type = params["neural_srp"]["metadata_type"]

        # 1. Create windowing transform
        self.window = Window(
            params["win_size"],
            int(params["win_size"] * params["hop_rate"]),
            window="hann"
        )

        # 2. Create feature extractor
        self.feature_extractor = GCC(params["win_size"], 
                                    tau_max=params["nb_gcc_bins"] // 2,
                                    transform="phat", concat_bins=False,
                                    center=True)

    def forward(self, x):
        x["signal"] = self.window(x["signal"])
        x["signal"] = self.feature_extractor(x["signal"]).unsqueeze(-1) # Add channel dimension
        
        # Subselect pairs for training, if required
        feature_pairs, idxs = self._sample_pairs(
            x,
            self.n_mic_pairs,
            self.mic_pair_sampling_mode,
        )

        x["signal"] = feature_pairs.transpose(1, 2)

        x["mic_pos"] = prepare_mic_pos(
            x["mic_pos"], idxs, mode=self.metadata_type
        )
        
        return x

    def _sample_pairs(self, x, n_pairs, mode="random"):
        """
        Subselect feature pairs from a batch of feature matrices.

        Args:
            x (dict): x["signal"].shape == (batch_size, n_frames, n_mics, n_mics, n_feature, n_channels)
                      x["mic_pos"] has the positions of the microphones 
            n_pairs: Number of GCC pairs to sample
            mode: "random", "first" or "all"

        Returns:
            (batch_size, n_frames, n_pairs, n_features, n_channels)
        """
        feature_matrix = x["signal"]
        mic_pos = x["mic_pos"]

        assert mode in ["all", "random", "first", "distinct_angles"]
        batch_size, n_frames, n_mics, _, n_features, n_channels = feature_matrix.shape

        idx_pairs = torch.zeros(
            (batch_size, n_pairs, 2), dtype=torch.long, device=feature_matrix.device,
        )

        idx_all_pairs = get_all_pairs(n_mics, device=feature_matrix.device)
        n_all_pairs = idx_all_pairs.shape[0]
        if mode == "first":
            idx_pairs = (
                torch.tensor(
                    [[0, i] for i in range(1, n_mics)],
                    dtype=torch.long,
                    device=feature_matrix.device,
                )
            )
        elif mode == "all":
            idx_pairs = idx_all_pairs
        elif mode == "random":
            perm = torch.randperm(n_all_pairs)[:n_pairs]
            idx_pairs = idx_all_pairs[perm]
        elif mode == "distinct_angles":
            idx_pairs = select_pairs(mic_pos[0])
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")
        
        idx_pairs = idx_pairs.unsqueeze(0).expand(batch_size, -1, -1)

        n_pairs = idx_pairs.shape[1]
        pairs = torch.zeros(
            (batch_size, n_frames, n_pairs, n_features, n_channels),
            device=feature_matrix.device, dtype=feature_matrix.dtype
        )

        for i in range(batch_size):
            # Select the corresponding GCCs
            pairs[i] = feature_matrix[i, :, idx_pairs[i, :, 0], idx_pairs[i, :, 1]]

        return pairs, idx_pairs
