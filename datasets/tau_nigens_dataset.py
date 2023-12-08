#
# Data generator for training the SELDnet
#

import os
import numpy as np
import random
import torch

from collections import deque
from datasets.mic_pos_utils import get_all_pairs, prepare_mic_pos

from datasets.preprocess_tau_nigens_dataset import Preprocessor 
from datasets.array_setup import TAU_NIGENS_TETRAHEDRAL


class TauNigensDataLoader(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False
    ):
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params['training']['batch_size']
        self._nb_gcc_bins = params['nb_gcc_bins']
        self._metadata_type = params['neural_srp']['metadata_type']

        hop_size = params['hop_rate']*params['win_size']
        self._feature_seq_len = int(params['fs'] * params['dataset']['tau_nigens']['duration_s']//hop_size)
        self._label_seq_len = params['dataset']['tau_nigens']['label_seq_len']
        self._shuffle = shuffle
        self._preprocessor = Preprocessor(params=params, is_eval=self._is_eval)
        self._label_dir = self._preprocessor.get_label_dir()
        self._feat_dir = self._preprocessor.get_feat_dir()
        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._nb_classes = self._preprocessor.get_nb_classes()
        self._get_filenames_list_and_feat_label_sizes()

        self._feature_batch_seq_len = int(self._batch_size*self._feature_seq_len)
        self._label_batch_seq_len = self._batch_size*self._label_seq_len
        self._circ_buf_feat = None
        self._circ_buf_label = None

        if self._per_file:
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor((len(self._filenames_list) * self._nb_frames_file /
                                               float(self._feature_batch_seq_len))))
        print(
            '\tnb_files: {}, nb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                len(self._filenames_list),
                self._nb_frames_file, self._nb_gcc_bins, self._nb_ch, self._label_len
                )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_gcc_bins)
        if self._is_eval:
            label_shape = None
        else:
            label_shape = [self._batch_size, self._label_seq_len, self._nb_classes*3]
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        for filename in os.listdir(self._feat_dir):
            if filename == ".DS_Store":
                # Skip mac specific file
                continue
            if int(filename[4]) in self._splits: # check which split the file belongs to
                self._filenames_list.append(filename)


        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))
        self._nb_frames_file = temp_feat.shape[0]
        self._nb_ch = temp_feat.shape[1] // self._nb_gcc_bins

        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
            self._label_len = temp_label.shape[-1]
            self._doa_len = (self._label_len - self._nb_classes)//self._nb_classes

        if self._per_file:
            self._batch_size = int(np.ceil(temp_feat.shape[0]/float(self._feature_seq_len)))

        return

    def get_batch(self):
        """
        Generates batches of samples
        :return: 
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)

        # Ideally this should have been outside the while loop. But while generating the test data we want the data
        # to be the same exactly for all epoch's hence we keep it here.
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()

        # Prepare metadata
        mic_pos = torch.from_numpy(
            TAU_NIGENS_TETRAHEDRAL['mic_pos']
        ).unsqueeze(0).repeat((self._batch_size, 1, 1))
        mic_pair_idxs = get_all_pairs(
            mic_pos.shape[1]
        ).unsqueeze(0).repeat((self._batch_size, 1, 1))
        mic_pairs = prepare_mic_pos(
            mic_pos, mic_pair_idxs,
            mode=self._metadata_type
        )

        file_cnt = 0

        for i in range(self._nb_total_batches):
            # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
            # circular buffer. If not keep refilling it.
            while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                for f_row in temp_feat:
                    self._circ_buf_feat.append(f_row)
                for l_row in temp_label:
                    self._circ_buf_label.append(l_row)

                # If self._per_file is True, this returns the sequences belonging to a single audio recording
                if self._per_file:
                    feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                    extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                    label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                    extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                    for f_row in extra_feat:
                        self._circ_buf_feat.append(f_row)
                    for l_row in extra_labels:
                        self._circ_buf_label.append(l_row)

                file_cnt = file_cnt + 1

            # Read one batch size from the circular buffer
            feat = np.zeros((self._feature_batch_seq_len, self._nb_gcc_bins * self._nb_ch))
            label = np.zeros((self._label_batch_seq_len, self._label_len))
            for j in range(self._feature_batch_seq_len):
                feat[j, :] = self._circ_buf_feat.popleft()
            for j in range(self._label_batch_seq_len):
                label[j, :] = self._circ_buf_label.popleft()
            feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_gcc_bins))

            # Split to sequences
            feat = self._split_in_seqs(feat, self._feature_seq_len)
            feat = np.transpose(feat, (0, 2, 1, 3))
            feat = torch.Tensor(feat)
            feat = feat.unsqueeze(-1) # Add channel dimension

            label = torch.Tensor(self._split_in_seqs(label, self._label_seq_len))

            # add batch dimension with the same number of batch samples as 'feat' to metadata
            # feat = _normalize_axis(feat, axis=-1) # Normalize along GCC axis

            feat = {'signal': feat, 'mic_pos': mic_pairs}

            yield feat, label

    def _split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._preprocessor.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._preprocessor.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._preprocessor.get_nb_frames()
    
    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._preprocessor.write_output_format_file(_out_file, _out_dict)

    def __len__(self):
        return self._nb_total_batches

def _normalize_axis(data, axis=-1):
    norm  = np.max(np.abs(data), axis=axis, keepdims=True)
    return data / (norm + 1e-8)
