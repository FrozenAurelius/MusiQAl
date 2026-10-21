import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


print("\n\n---------------------- extract_dinov2_feats -------------------\n\n")

# 224 / 14 = 16x16 patches
H, W, PATCH = 224, 224, 14


def TransformImage(img):

    # imagenet mean/std, same as dinov2 training
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trans = transforms.Compose([
        transforms.Resize([H, W]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    frame_tensor = trans(img)

    return frame_tensor


def load_frame_info(img_path):

    img = Image.open(img_path).convert('RGB')
    frame_tensor = TransformImage(img)

    return frame_tensor


def load_model(name):

    # compute nodes have no internet, so use the cached copy if it is there
    local_repo = os.path.join(torch.hub.get_dir(), 'facebookresearch_dinov2_main')
    if os.path.isdir(local_repo):
        model = torch.hub.load(local_repo, name, source='local')
    else:
        model = torch.hub.load('facebookresearch/dinov2', name)

    return model


def extract_feats(params, model):

    model.eval()
    dir_fc = os.path.join(os.getcwd(), params['output_dir'])
    if not os.path.isdir(dir_fc):
        os.makedirs(dir_fc)

    video_list = sorted(os.listdir(params['video_path']))
    list_len = len(video_list)
    print(list_len)

    cnt = 0
    for video in video_list:

        print("name: ", video)
        outfile = os.path.join(dir_fc, video + '.npy')
        if os.path.exists(outfile):
            print(video, " is already exist!")
            continue

        image_list = sorted(glob.glob(os.path.join(params['video_path'], video, '*.jpg')))
        print("img len: ", len(image_list))

        # always 60 frames, the dataloader expects this
        samples = np.round(np.linspace(0, len(image_list) - 1, params['n_frames'])).astype(int)
        image_list = [image_list[i] for i in samples]

        select_img = torch.stack([load_frame_info(img) for img in image_list])     # [60, 3, 224, 224]

        with torch.no_grad():
            out = model.forward_features(select_img.cuda())

        # only the patch tokens, no cls / register tokens
        fea = out['x_norm_patchtokens']                                 # [60, 256, 768]
        T, N, C = fea.shape
        fea = fea.reshape(T, H // PATCH, W // PATCH, C).permute(0, 3, 1, 2)   # [60, 768, 16, 16]
        fea = fea.cpu().numpy()

        if params['fp16']:
            fea = fea.astype(np.float16)

        np.save(outfile, fea)

        cnt = cnt + 1
        print("----------------->> ", cnt, " / ", list_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", dest='output_dir', type=str, default='./AVST/data/feats/dinov2_vitb14_reg/',
                        help='directory to store features')
    parser.add_argument("--n_frames", dest='n_frames', type=int, default=60,
                        help='how many frames to sample per video')
    parser.add_argument("--video_path", dest='video_path', type=str, default='./AVST/data/frames',
                        help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='dinov2_vitb14_reg',
                        help='dinov2 model from torch hub')
    parser.add_argument("--fp16", dest='fp16', action='store_true',
                        help='save features as float16')

    args = parser.parse_args()
    params = vars(args)

    model = load_model(params['model']).cuda()
    extract_feats(params, model)
