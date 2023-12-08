# Contains routines for labels creation, features extraction and normalization
#

import json
import joblib
import matplotlib.pyplot as plot
import math
import numpy as np
import os
import scipy.io.wavfile as wav

from sklearn import preprocessing

from models.numpy_transforms import GccExtractor
from utils import create_folder

plot.switch_backend('agg')


def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


class Preprocessor:
    def __init__(self, params, is_eval=False, gcc_mode='first'):
        """
        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """

        self._gcc_extractor = GccExtractor(params, gcc_mode=gcc_mode)

        # Input directories
        self._feat_label_dir = params['path_tau_nigens_preprocessed']
        self._dataset_dir = params['path_tau_nigens']
        self._dataset_combination = 'mic_{}'.format('eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params['fs']
        self._hop_len = int(params["hop_rate"]*params["win_size"])
        self._hop_len_s = self._hop_len/params["fs"]
        
        self._label_hop_len_s = params["dataset"]['tau_nigens']["label_hop_len_s"]
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._dataset = 'mic'
        self._eps = 1e-8
        self._nb_channels = 4

        # Sound event classes dictionary
        self._nb_unique_classes = 2
        self._audio_max_len_samples = params['dataset']['tau_nigens']['max_audio_len_s'] * self._fs  # TODO: Fix the audio synthesis code to always generate 60s of
        # audio. Currently it generates audio till the last active sound event, which is not always 60s long. This is a
        # quick fix to overcome that. We need this because, for processing and training we need the length of features
        # to be fixed.

        self._max_feat_frames = int(self._audio_max_len_samples / float(self._hop_len))
        self._max_label_frames = int(self._audio_max_len_samples / float(self._label_hop_len))

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.random.rand(self._audio_max_len_samples - audio.shape[0], audio.shape[1])*self._eps
            audio = np.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio, fs

    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)

        x_label = 10*np.ones((self._max_label_frames, self._nb_unique_classes))
        y_label = 10*np.ones((self._max_label_frames, self._nb_unique_classes))
        z_label = 10*np.ones((self._max_label_frames, self._nb_unique_classes))

        nb_classes = np.zeros((self._max_label_frames, self._nb_unique_classes))
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self._max_label_frames:
                for active_event in range(len(active_event_list)):
                    nb_classes[frame_ind, active_event] = 1
                    x_label[frame_ind, active_event] = active_event_list[active_event][2]
                    y_label[frame_ind, active_event] = active_event_list[active_event][3]
                    z_label[frame_ind, active_event] = active_event_list[active_event][4]
        label_mat = np.concatenate((x_label, y_label, z_label, nb_classes), axis=1)

        return label_mat

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)

        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))

        for file_cnt, file_name in enumerate(os.listdir(self._aud_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            audio_in, fs = self._load_audio(os.path.join(self._aud_dir, wav_filename))

            #extract gcc features
            feat = self._gcc_extractor(audio_in)
            nb_frames = feat.shape[1]
            feat = feat.transpose((0, 2, 1)).reshape((nb_frames, -1))

            print('{}: {}, {}'.format(file_cnt, file_name, feat.shape))
            np.save(os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0])), feat)

    def _init_scaler(self):
        normalized_features_wts_file = self.get_normalized_wts_file()
        
        self._spec_scaler = joblib.load(normalized_features_wts_file)
        print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        return self._spec_scaler

    def scale_features(self, feat):
        if len(feat.shape) == 3:
            nb_mic_pairs, nb_frames, nb_bins = feat.shape 
            feat = feat.transpose((0, 2, 1)).reshape((nb_frames, -1))
            feat = self._spec_scaler.transform(feat)
            feat = feat.reshape((nb_mic_pairs, nb_bins, nb_frames)).transpose((0, 2, 1))
        elif len(feat.shape) == 2:
            feat = self._spec_scaler.transform(feat)
        else:
            raise ValueError('Feature shape not supported: {}'.format(feat.shape))
        return feat
        
    def preprocess_features(self):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()

        # pre-processing starts
        if self._is_eval:
            self.init_scaler()
        else:
            print('Estimating weights for normalizing feature files:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                spec_scaler.partial_fit(feat_file)
                del feat_file
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )
            self._spec_scaler = spec_scaler
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            features = np.load(os.path.join(self._feat_dir, file_name))
            features = self.scale_features(features)
            np.save(
                os.path.join(self._feat_dir_norm, file_name),
                features
            )
            del features

        print('normalized files written to {}'.format(self._feat_dir_norm))

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])

            try:
                desc_file_polar = self.load_output_format_file(os.path.join(self._desc_dir, file_name))
            except Exception as e:
                print("Error while loading file: {}".format(file_name))
                continue

            desc_file = self.convert_output_format_spherical_to_cartesian(desc_file_polar)
            label_mat = self.get_labels_for_file(desc_file)
            print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
            np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)
    
    def load_output_format_file(self, _output_format_file):
        """
        Loads DCASE output format csv file and returns it in dictionary format
        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 5: #polar coordinates 
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            elif len(_words) == 6: # cartesian coordinates
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
        _fid.close()
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary
        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count we use a fixed value.
                _fid.write('{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3])))
        _fid.close()

    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames/float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt+self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict

    def regression_label_format_to_output_format(self, _sed_labels, _doa_labels):
        """
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.
        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _nb_classes = self._nb_unique_classes
        _is_polar = _doa_labels.shape[-1] == 2*_nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2*_nb_classes]
            _z = _doa_labels[:, 2*_nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append([_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append([_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class], _z[_frame_ind, _tmp_class]])
        return _output_dict

    def convert_output_format_spherical_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:

                    ele_rad = tmp_val[3]*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
        return out_dict

    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                    # in degrees
                    azimuth = np.arctan2(y, x) * 180 / np.pi
                    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                    r = np.sqrt(x**2 + y**2 + z**2)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], azimuth, elevation])
        return out_dict

    # ------------------------------- Misc public functions -------------------------------

    def get_feat_dir(self):
        norm_path = os.path.join(
            self._feat_label_dir,
            '{}_norm'.format(self._dataset_combination))
        if os.path.exists(norm_path):
            return norm_path
        else:
            return os.path.join(
                self._feat_label_dir,
                self._dataset_combination
            )
    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format(self._dataset_combination)
        )

    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format(self._dataset_combination)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir, '{}_label'.format(self._dataset_combination)
            )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return self._nb_unique_classes

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_frames(self):
        return self._max_label_frames

    def get_nb_bins(self):
        return self._gcc_extractor._nb_bins


if __name__ == "__main__":
    
    params = json.load(open('params.json', 'r'))

    # -------------- Extract features and labels for development set -----------------------------
    preprocessor = Preprocessor(params, gcc_mode='all')

    # # Extract features and normalize them
    preprocessor.extract_all_feature()
    preprocessor.preprocess_features()

    # # Extract labels in regression mode
    preprocessor.extract_all_labels()