import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class AVQA_grd_feat_dataset(Dataset):
    # one sample = one second of audio, its own frame (match, label 1)
    # and a frame from another video (no match, label 0).
    # uses all 60 seconds of each video, read from the saved features

    def __init__(self, video_list, audio_dir, video_feat_dir, mode_flag='train'):

        self.video_list = video_list
        self.audio_dir = audio_dir
        self.video_feat_dir = video_feat_dir
        self.mode_flag = mode_flag
        self.n_frames = 60

    def __len__(self):
        return self.n_frames * len(self.video_list)

    def load_frame(self, video_name, t):
        # mmap so only frame t is read, dinov2 feats are saved as float16
        feat = np.load(os.path.join(self.video_feat_dir, video_name + '.npy'), mmap_mode='r')
        return torch.from_numpy(np.array(feat[t], dtype=np.float32))

    def __getitem__(self, idx):

        pos_video_id = idx // self.n_frames
        t = idx % self.n_frames
        pos_video_name = self.video_list[pos_video_id]

        # val: the same negatives every epoch
        rng = random if self.mode_flag == 'train' else random.Random(idx)
        neg_video_id = rng.randrange(len(self.video_list) - 1)
        if neg_video_id >= pos_video_id:
            neg_video_id += 1
        neg_t = rng.randrange(self.n_frames)
        neg_video_name = self.video_list[neg_video_id]

        pos_frame = self.load_frame(pos_video_name, t)
        neg_frame = self.load_frame(neg_video_name, neg_t)
        sec_audio = torch.from_numpy(np.load(os.path.join(self.audio_dir, pos_video_name + '.npy'))[t].astype(np.float32))

        video_s = torch.stack((pos_frame, neg_frame))       # [2, C, H, W]
        audio = torch.stack((sec_audio, sec_audio))         # [2, 128]
        label = torch.tensor([1, 0])

        return {'audio': audio, 'video_s': video_s, 'label': label}
