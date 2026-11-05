# Compares the saved ResNet-18 and DINOv2 features directly, before AVST sees them.
#
#   python AVST/tools/feature_viz.py
#
# 1. separability: per-video labels taken from the videos' own questions (performance type,
#    indoor/outdoor, number of instruments heard). Leave-one-out nearest-neighbour accuracy
#    on the video's mean feature, and 2D maps (PCA by default, t-SNE with --method tsne).
# 2. patch maps: the first 3 PCA components of the patch features as RGB over the frame,
#    to see whether patches separate performers / instruments from the background.
# 3. change over time: cosine similarity between the frames AVST uses (6 s apart) in one video,
#    against frames from different videos.
#
# Features are pooled over the grid as in AVST (avgpool), using the 10 frames AVST uses.
import os
import json
import argparse
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKBONES = [['ResNet-18', 'res18_14x14'], ['DINOv2', 'dinov2_vitb14_reg']]

# label name, question text, answer -> class (None = drop the video)
INSTR = {'zero': '0', 'one': '1', 'two': '2', 'three': '3'}
LABELS = [
    ['performance type', 'What type of performance is happening in this video? ', lambda a: a],
    ['indoor / outdoor', 'Where is the performance? ', lambda a: a],
    ['instruments heard', 'How many musical instruments were heard throughout the video?', lambda a: INSTR.get(a, '4+')],
]

COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75)]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:     # older Pillow
        return ImageFont.load_default()


def video_labels(json_files):
    # one label per video and question type, from all three splits (a video's questions are spread over the splits)
    qs = []
    for f in json_files:
        qs += json.load(open(f))
    out = {}
    for name, text, fn in LABELS:
        out[name] = {q['video_id']: fn(q['answer']) for q in qs if q['question_content'] == text}
    return out


def load_frames(feat_dir, video):
    # [10, C, H, W] float32, the frames AVST uses
    f = np.load(os.path.join(feat_dir, video + '.npy'), mmap_mode='r')
    return np.ascontiguousarray(f[::6], dtype=np.float32)


def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def loo_knn(X, y, k=5):
    # leave-one-out k-nearest-neighbour accuracy, cosine similarity, majority vote
    Xn = l2n(X)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    nn_idx = np.argsort(-S, axis=1)[:, :k]
    correct = 0
    for i in range(len(y)):
        votes = Counter(y[j] for j in nn_idx[i])
        correct += votes.most_common(1)[0][0] == y[i]
    return 100 * correct / len(y)


def embed_2d(X, method='pca', seed=0):
    Xn = l2n(X - X.mean(0))
    # PCA (via SVD): the first 2 components are the plot; 50 components are the input for t-SNE
    U, S, _ = np.linalg.svd(Xn - Xn.mean(0), full_matrices=False)
    P = U[:, :50] * S[:50]
    if method == 'tsne':
        from sklearn.manifold import TSNE   # needs scikit-learn
        return TSNE(n_components=2, perplexity=30, init='pca', random_state=seed).fit_transform(P), 't-SNE'
    return P[:, :2], 'PCA'


def scatter(ax_size, pts, y, classes, title):
    W, H = ax_size
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    d.text((W // 2, 14), title, fill='black', font=font(16), anchor='mm')
    pad, top = 20, 34
    lo, hi = pts.min(0), pts.max(0)
    p = (pts - lo) / (hi - lo + 1e-8)
    for (px, py), lab in zip(p, y):
        x = pad + px * (W - 2 * pad)
        yy = top + (1 - py) * (H - top - pad - 22)
        c = COLORS[classes.index(lab) % len(COLORS)]
        d.ellipse((x - 4, yy - 4, x + 4, yy + 4), fill=c)
    # legend along the bottom
    x = pad
    for i, c in enumerate(classes):
        d.ellipse((x, H - 16, x + 9, H - 7), fill=COLORS[i % len(COLORS)])
        d.text((x + 13, H - 12), c, fill='black', font=font(13), anchor='lm')
        x += 22 + 8 * len(c)
    d.rectangle((0, 0, W - 1, H - 1), outline=(200, 200, 200))
    return img


def patch_pca_rgb(frames_feat):
    # frames_feat: [N, C, H, W] for the frames shown -> [N, H, W, 3] in [0, 1]
    N, C, H, W = frames_feat.shape
    X = frames_feat.transpose(0, 2, 3, 1).reshape(-1, C)
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:3].T
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    P = np.clip((P - lo) / (hi - lo + 1e-8), 0, 1)
    return P.reshape(N, H, W, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', default='AVST/data/feats/')
    parser.add_argument('--frames_dir', default='AVST/data/frames')
    parser.add_argument('--json', nargs='+', default=['AVST/data/json/avqa-train.json', 'AVST/data/json/avqa-val.json', 'AVST/data/json/avqa-test.json'])
    parser.add_argument('--videos', nargs='+', default=['00000155', '00000034', '00000000', '00000100'], help='videos for the patch maps')
    parser.add_argument('--seconds', type=int, nargs='+', default=[0, 30])
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--method', choices=['pca', 'tsne'], default='pca', help='2D projection for the plots (the paper uses PCA)')
    parser.add_argument('--out_dir', default='AVST/feature_viz')
    parser.add_argument('--tile_width', type=int, default=280)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    backbones = [b for b in BACKBONES if os.path.isdir(os.path.join(args.feat_dir, b[1]))]
    print('backbones:', [b[0] for b in backbones])
    labels = video_labels(args.json)
    videos = sorted(os.listdir(os.path.join(args.feat_dir, backbones[0][1])))
    videos = [v[:-4] for v in videos if v.endswith('.npy')]
    print('videos:', len(videos))

    # ---------------------------------------------------- per-video features
    feats = {}       # backbone -> [V, 10, C] grid-pooled frames
    for name, d in backbones:
        feats[name] = np.stack([load_frames(os.path.join(args.feat_dir, d), v).mean(axis=(2, 3)) for v in videos])
        print(name, feats[name].shape)

    # ---------------------------------------------------- 1. separability
    print('\n1. leave-one-out %d-NN accuracy (%%) on the video mean feature' % args.k)
    print('%-20s %5s %9s' % ('label', 'n', 'majority') + ''.join('%12s' % n for n, _ in backbones))
    tiles = []
    for lname, _, _ in LABELS:
        idx = [i for i, v in enumerate(videos) if v in labels[lname]]
        y = [labels[lname][videos[i]] for i in idx]
        classes = sorted(set(y))
        majority = 100 * Counter(y).most_common(1)[0][1] / len(y)
        row = '%-20s %5d %9.1f' % (lname, len(y), majority)
        for bname, _ in backbones:
            X = feats[bname][idx].mean(1)
            row += '%12.1f' % loo_knn(X, y, args.k)
        print(row)
        row_tiles = []
        for bname, _ in backbones:
            pts, method = embed_2d(feats[bname][idx].mean(1), args.method)
            row_tiles.append(scatter((420, 380), pts, y, classes, '%s · %s (%s)' % (bname, lname, method)))
        tiles.append(row_tiles)
    fig = Image.new('RGB', (420 * len(backbones), 380 * len(tiles)), 'white')
    for i, row in enumerate(tiles):
        for j, t in enumerate(row):
            fig.paste(t, (j * 420, i * 380))
    fig.save(os.path.join(args.out_dir, 'embeddings.png'))
    print('saved', os.path.join(args.out_dir, 'embeddings.png'))

    # ---------------------------------------------------- 3. change over time
    print('\n3. mean cosine similarity between grid-pooled frames')
    print('%-12s %22s %22s %22s' % ('backbone', 'next frame (6 s)', 'same video, any', 'different videos'))
    rng = np.random.default_rng(0)
    for bname, _ in backbones:
        F = l2n(feats[bname])                                  # [V, 10, C]
        nxt = (F[:, 1:] * F[:, :-1]).sum(-1).mean()
        within = np.einsum('vtc,vsc->vts', F, F)
        within = within[:, ~np.eye(10, dtype=bool)].mean()
        a, b = rng.integers(0, len(videos), 5000), rng.integers(0, len(videos), 5000)
        keep = a != b
        ta, tb = rng.integers(0, 10, 5000), rng.integers(0, 10, 5000)
        between = (F[a[keep], ta[keep]] * F[b[keep], tb[keep]]).sum(-1).mean()
        print('%-12s %22.3f %22.3f %22.3f' % (bname, nxt, within, between))

    # ---------------------------------------------------- 2. patch maps
    if not os.path.isdir(args.frames_dir):
        print('\nno frames in', args.frames_dir, '- skipping the patch maps')
        return
    for s in args.seconds:
        assert s % 6 == 0 and 0 <= s < 60, 'AVST only sees seconds 0, 6, ..., 54'
    tw = args.tile_width
    for video in args.videos:
        frames = sorted(f for f in os.listdir(os.path.join(args.frames_dir, video)) if f.endswith('.jpg'))
        fidx = np.round(np.linspace(0, len(frames) - 1, 60)).astype(int)
        imgs = []
        for s in args.seconds:
            im = Image.open(os.path.join(args.frames_dir, video, frames[fidx[s]])).convert('RGB')
            imgs.append(im.resize((tw, round(im.size[1] * tw / im.size[0]))))
        th = imgs[0].size[1]
        rows = [('Frame', imgs)]
        for bname, d in backbones:
            f = np.load(os.path.join(args.feat_dir, d, video + '.npy'), mmap_mode='r')
            rgb = patch_pca_rgb(np.stack([np.asarray(f[s], dtype=np.float32) for s in args.seconds]))
            rows.append((bname, [Image.fromarray((m * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST) for m in rgb]))
        left, top = 110, 26
        fig = Image.new('RGB', (left + tw * len(args.seconds), top + th * len(rows)), 'white')
        d = ImageDraw.Draw(fig)
        for j, s in enumerate(args.seconds):
            d.text((left + j * tw + tw // 2, top // 2), '%d s' % s, fill='black', font=font(16), anchor='mm')
        for i, (lab, row) in enumerate(rows):
            d.text((8, top + i * th + th // 2), lab, fill='black', font=font(16), anchor='lm')
            for j, t in enumerate(row):
                fig.paste(t, (left + j * tw, top + i * th))
        out = os.path.join(args.out_dir, 'patches_%s.png' % video)
        fig.save(out)
        print('saved', out)


if __name__ == '__main__':
    main()
