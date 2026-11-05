# Draws where AVST looks for the sound: the audio-visual grounding weights over the image grid,
# for the trained models of each setup, on top of the video frames.
#
#   python AVST/tools/attention_maps.py --videos 00000155 00000034
#
# The map does not depend on the question, only on the audio and the frame of that second.
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'net_grd_avst'))
from net_avst import AVQA_Fusion_Net

MODELS = 'AVST/net_grd_avst/avst_models/'
FEATS = 'AVST/data/feats/'

# label, checkpoint, feature dir, visual_dim, temporal_layers
SETUPS = [
    ['A: ResNet-18', 'avst_seed1', 'res18_14x14', '512', '0'],
    ['B: DINOv2', 'avst_dinov2_seed1', 'dinov2_vitb14_reg', '768', '0'],
    ['C: + grounding', 'avst_dinov2_grd_seed1', 'dinov2_vitb14_reg', '768', '0'],
    ['D: + temporal', 'avst_dinov2_grd_temp_seed1', 'dinov2_vitb14_reg', '768', '2'],
]


def load_model(ckpt, visual_dim, temporal_layers, device):
    model = AVQA_Fusion_Net(visual_dim=visual_dim, temporal_layers=temporal_layers)
    # main_avst.py saves the DataParallel-wrapped model, so the keys start with 'module.'
    sd = torch.load(ckpt, map_location='cpu')
    model.load_state_dict({k.replace('module.', '', 1): v for k, v in sd.items()})
    return model.to(device).eval()


@torch.no_grad()
def sound_attention(model, audio, visual):
    '''same steps as the posi grounding in AVQA_Fusion_Net.forward
        audio:  [T, 128]
        visual: [T, C, H, W]
        returns the softmax weights over the grid, [T, H, W]
    '''
    visual = visual.unsqueeze(0)
    if model.fc_visual is not None:
        visual = model.fc_visual(visual.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3).contiguous()
    if model.temporal is not None:
        visual = model.temporal_encode(visual)
    _, T, C, H, W = visual.size()

    audio_feat = model.fc_a2(F.relu(model.fc_a1(audio)))                     # [T, 512]
    audio_feat = nn.functional.normalize(audio_feat, dim=1).unsqueeze(-1)    # [T, 512, 1]
    v_feat = visual.view(T, C, H * W).permute(0, 2, 1)                      # [T, HxW, 512]
    v_feat = nn.functional.normalize(v_feat, dim=2)

    x2_va = torch.matmul(v_feat, audio_feat).squeeze(-1)                     # [T, HxW]
    return F.softmax(x2_va, dim=-1).view(T, H, W)


def colormap(x):
    # x in [0, 1] -> blue, cyan, yellow, red
    stops = np.array([[0, 0, 140], [0, 200, 255], [255, 230, 0], [220, 0, 0]], dtype=np.float32)
    pos = x * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    t = (pos - i)[..., None]
    return (stops[i] * (1 - t) + stops[i + 1] * t).astype(np.uint8)


def overlay(frame, att, alpha=0.5):
    # grid weights -> image size, scaled per frame to [0, 1]
    m = Image.fromarray(att.astype(np.float32), mode='F').resize(frame.size, Image.BILINEAR)
    m = np.asarray(m)
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    heat = Image.fromarray(colormap(m))
    return Image.blend(frame, heat, alpha)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:     # older Pillow
        return ImageFont.load_default()


def grid_figure(rows, labels, col_labels, tile_w):
    tile_h = rows[0][0].size[1]
    left, top = int(tile_w * 0.55), 28
    fig = Image.new('RGB', (left + tile_w * len(rows[0]), top + tile_h * len(rows)), 'white')
    draw = ImageDraw.Draw(fig)
    for j, c in enumerate(col_labels):
        draw.text((left + j * tile_w + tile_w // 2, top // 2), c, fill='black', font=font(18), anchor='mm')
    for i, (row, label) in enumerate(zip(rows, labels)):
        draw.text((8, top + i * tile_h + tile_h // 2), label, fill='black', font=font(18), anchor='lm')
        for j, tile in enumerate(row):
            fig.paste(tile, (left + j * tile_w, top + i * tile_h))
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', nargs='+', default=None, help='video ids, default: the first test videos with Audio-Visual Location questions')
    parser.add_argument('--n_videos', type=int, default=4)
    parser.add_argument('--seconds', type=int, nargs='+', default=[0, 18, 36, 54], help='seconds to show, AVST uses 0, 6, ..., 54')
    parser.add_argument('--setup', nargs=5, action='append', default=None,
                        metavar=('LABEL', 'CHECKPOINT', 'FEATS', 'VDIM', 'TLAYERS'), help='replaces the default four setups')
    parser.add_argument('--model_dir', default=MODELS)
    parser.add_argument('--feat_dir', default=FEATS)
    parser.add_argument('--audio_dir', default=FEATS + 'vggish')
    parser.add_argument('--frames_dir', default='AVST/data/frames')
    parser.add_argument('--label_test', default='AVST/data/json/avqa-test.json')
    parser.add_argument('--out_dir', default='AVST/attention_maps')
    parser.add_argument('--tile_width', type=int, default=320)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setups = args.setup or SETUPS
    for s in args.seconds:
        assert s % 6 == 0 and 0 <= s < 60, 'AVST only sees seconds 0, 6, ..., 54'

    videos = args.videos
    if videos is None:
        videos = []
        for q in json.load(open(args.label_test)):
            if json.loads(q['type']) == ['Audio-Visual', 'Location'] and q['video_id'] not in videos:
                videos.append(q['video_id'])
        videos = videos[:args.n_videos]
    print('videos:', videos)

    models = [(label, load_model(os.path.join(args.model_dir, ckpt + '.pt'), int(vdim), int(tl), device), feats)
              for label, ckpt, feats, vdim, tl in setups]
    os.makedirs(args.out_dir, exist_ok=True)
    focus = {}                       # label -> all max-weight ratios, for the summary at the end

    for video in videos:
        audio = torch.from_numpy(np.load(os.path.join(args.audio_dir, video + '.npy'))[::6].astype(np.float32)).to(device)
        frames = sorted(f for f in os.listdir(os.path.join(args.frames_dir, video)) if f.endswith('.jpg'))
        # the feature scripts sample 60 frames evenly, so second t is frame index t
        idx = np.round(np.linspace(0, len(frames) - 1, 60)).astype(int)

        images = []
        for s in args.seconds:
            img = Image.open(os.path.join(args.frames_dir, video, frames[idx[s]])).convert('RGB')
            images.append(img.resize((args.tile_width, round(img.size[1] * args.tile_width / img.size[0]))))

        rows, labels = [images], ['Frame']
        for label, model, feats in models:
            visual = np.load(os.path.join(args.feat_dir, feats, video + '.npy'), mmap_mode='r')
            visual = torch.from_numpy(np.ascontiguousarray(visual[::6], dtype=np.float32)).to(device)
            att = sound_attention(model, audio, visual).cpu().numpy()             # [10, H, W]
            n = att.shape[1] * att.shape[2]
            for s in args.seconds:
                # how peaked the weights are: 1.0 would mean the same weight everywhere
                print(f'{video} {label:16s} second {s:2d}: max weight {att[s // 6].max() * n:.2f} x uniform')
                focus.setdefault(label, []).append(float(att[s // 6].max() * n))
            rows.append([overlay(img, att[s // 6]) for img, s in zip(images, args.seconds)])
            labels.append(label)

        fig = grid_figure(rows, labels, [f'{s} s' for s in args.seconds], args.tile_width)
        out = os.path.join(args.out_dir, video + '.png')
        fig.save(out)
        print('saved', out)

    # the numbers in the paper: mean and range of the max-weight ratio per setup, over all videos and seconds
    print('\nlargest weight / uniform weight, per setup:')
    for label, vals in focus.items():
        print(f'{label:16s} n={len(vals):3d}  mean {np.mean(vals):.2f}  min {min(vals):.2f}  max {max(vals):.2f}')


if __name__ == '__main__':
    main()
