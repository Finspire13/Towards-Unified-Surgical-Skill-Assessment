import os
import copy
import random
import numpy as np
import skvideo.io
import torch
from torch.utils.data import Dataset


def get_data_dict(video_dir, label_dir, feature_dir_list,
                  video_list, score_key_list, score_range_list,
                  new_sample_rate=1, old_sample_rate=1,
                  frame_rate=30, temporal_aug=True, data_check=False):
    # [TO BE IMPROVED] default snippet_size=1,
    # leave strict length check to other codes, here only loosely check and pad

    assert (new_sample_rate > 0)
    assert (old_sample_rate > 0)
    assert (new_sample_rate % old_sample_rate == 0)
    f_sample_rate = int(new_sample_rate / old_sample_rate)  # sample_rate of feature sequences

    data_dict = {k: {
        'duration': None,
        'num_frames': None,
        'frame_rate': frame_rate,
        'feature_list': None,
        'score': None,
    } for k in video_list
    }

    for video in video_list:

        feature_files = [os.path.join(feature_dir, '{}.npy'.format(video))
                         for feature_dir in feature_dir_list]

        score_file = [i for i in os.listdir(label_dir)
                      if 'FT-score' in i and 'npy' in i and video in i]
        assert (len(score_file) == 1)
        score_file = os.path.join(label_dir, score_file[0])

        score = np.load(score_file, allow_pickle=True).item()
        score = [(score[score_key_list[i]] - score_range_list[i][0]) / (score_range_list[i][1] - score_range_list[i][0])
                 for i in range(len(score_key_list))]
        assert (np.array(score).max() <= 1)
        assert (np.array(score).min() >= 0)

        print('Loading Feature List: {}'.format(video))

        feature_list = [np.swapaxes(np.load(feature_file, allow_pickle=True), 0, 1)
                        for feature_file in feature_files]

        print('Feature List Loaded: {} - {}'.format(video, [i.shape for i in feature_list]))

        if data_check:  # Length Check

            video_file = os.path.join(video_dir, '{}.mp4'.format(video))
            video_meta = skvideo.io.ffprobe(video_file)
            duration = float(video_meta['video']['@duration'])
            num_frames = int(video_meta['video']['@nb_frames'])
            frame_rate = num_frames / duration
            assert (np.abs(data_dict[video]['frame_rate'] - frame_rate) < 1e-6)

            for i in range(len(feature_list)):
                assert (feature_list[i].shape[1] == np.arange(num_frames)[::old_sample_rate].shape[0])

            data_dict[video]['duration'] = duration
            data_dict[video]['num_frames'] = num_frames

        # Temporal Augmentation
        for i in range(len(feature_list)):
            if temporal_aug:
                feature_list[i] = [feature_list[i][:, f_offset::f_sample_rate, :]
                                   for f_offset in range(f_sample_rate)]
            else:
                feature_list[i] = [feature_list[i][:, ::f_sample_rate, :]]

        data_dict[video]['feature_list'] = feature_list  # list of list
        data_dict[video]['score'] = score  # list

    return data_dict


class SurgeryFeatureDataset(Dataset):
    def __init__(self, data_dict, mode):

        super(SurgeryFeatureDataset, self).__init__()

        assert (mode in ['train', 'test'])

        self.data_dict = data_dict
        self.mode = mode
        self.video_list = [i for i in self.data_dict.keys()]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]

        feature_list = copy.deepcopy(self.data_dict[video]['feature_list'])  # Do not change data_dict
        score = np.array(self.data_dict[video]['score'])

        num_feature_types = len(feature_list)

        if self.mode == 'train':

            num_temporal_augs = len(feature_list[0])
            temporal_rid = random.randint(0, num_temporal_augs - 1)  # a<=x<=b

            for i in range(num_feature_types):
                num_spatial_augs = feature_list[i][0].shape[0]  # different for each feature type
                spatial_rid = random.randint(0, num_spatial_augs - 1)  # a<=x<=b

                feature_list[i] = feature_list[i][temporal_rid][spatial_rid].T
                # Now feature_list: list of same length sequences, [(F, T)]

            feature_list = [torch.from_numpy(i).float() for i in feature_list]

        if self.mode == 'test':

            for i in range(num_feature_types):
                feature_list[i] = [torch.from_numpy(
                    np.swapaxes(i, 1, 2)).float() for i in feature_list[i]]
                # Now feature_list: list of list of same length sequences, [[10 x F x T]]

        return feature_list, score, video
