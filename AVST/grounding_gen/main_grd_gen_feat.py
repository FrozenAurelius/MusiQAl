from __future__ import print_function
import sys
sys.path.append("./AVST/")

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from grounding_gen.dataloader_grd_gen_feat import AVQA_grd_feat_dataset
from grounding_gen.nets_grd_gen_feat import AVQA_AVatt_Grounding_Feat


print("\n ------------------- grounding_gen on saved feats ---------------------\n")

# layers shared with the original grounding stage (trained on resnet18 frames)
SHARED = ['fc_a1', 'fc_a2', 'fc_gl', 'fc1', 'fc2', 'fc3', 'fc4']


def train(args, model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        audio, video, target = sample['audio'].to(device), sample['video_s'].to(device), sample['label'].to(device)
        optimizer.zero_grad()

        out = model(audio, video)                   # [B*2, 2]
        loss = criterion(out, target.view(-1))      # [1, 0, 1, 0, ...]
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for sample in val_loader:
            audio, video, target = sample['audio'].to(device), sample['video_s'].to(device), sample['label'].to(device)
            _, predicted = torch.max(model(audio, video), 1)
            total += predicted.size(0)
            correct += (predicted == target.view(-1)).sum().item()

    print('Match accuracy: %.2f %%' % (100 * correct / total))
    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Grounding stage on saved visual features')

    parser.add_argument(
        "--audio_dir", type=str, default='./AVST/data/feats/vggish', help="audio dir")
    parser.add_argument(
        "--video_feat_dir", type=str, default='./AVST/data/feats/dinov2_vitb14_reg', help="saved visual feats")
    parser.add_argument(
        "--visual_dim", type=int, default=768, help="feature size per cell: 512 for ResNet-18, 768 for DINOv2")
    parser.add_argument(
        "--label_train", type=str, default="./AVST/data/json/avqa-train.json", help="videos to use")
    parser.add_argument(
        "--init_ckpt", type=str, default='./AVST/grounding_gen/models_grounding_gen/main_grounding_gen_best.pt',
        help="start from this grounding stage (resnet18 version), '' for random")
    parser.add_argument(
        '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument(
        '--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument(
        '--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument(
        '--num_workers', type=int, default=6, help='dataloader workers')
    parser.add_argument(
        "--model_save_dir", type=str, default='./AVST/grounding_gen/models_grounding_gen/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='grounding_dinov2', help="save model name")

    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AVQA_AVatt_Grounding_Feat(visual_dim=args.visual_dim)
    model = nn.DataParallel(model)
    model = model.to(device)

    # start from the resnet18 grounding stage, only fc_visual is new
    if args.init_ckpt:
        checkpoint = torch.load(args.init_ckpt, map_location='cpu')
        pretrained_dict = {k: v for k, v in checkpoint.items() if k.split('.')[1] in SHARED}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("from", args.init_ckpt, ":", sorted(pretrained_dict))

    # all questions use the same 310 videos, so hold out every 10th video for picking the best epoch
    video_list = []
    for sample in json.load(open(args.label_train, 'r')):
        if sample['video_id'] not in video_list:
            video_list.append(sample['video_id'])
    val_videos = video_list[::10]
    train_videos = [v for v in video_list if v not in val_videos]
    print("train videos:", len(train_videos), "val videos:", len(val_videos))

    train_dataset = AVQA_grd_feat_dataset(train_videos, args.audio_dir, args.video_feat_dir, mode_flag='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataset = AVQA_grd_feat_dataset(val_videos, args.audio_dir, args.video_feat_dir, mode_flag='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.model_save_dir, exist_ok=True)
    save_path = os.path.join(args.model_save_dir, args.checkpoint + "_best.pt")

    # epoch 0: how well the starting weights match audio and frames
    best_F = eval(model, val_loader, device)
    torch.save(model.state_dict(), save_path)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, criterion, epoch, device)
        scheduler.step()
        F = eval(model, val_loader, device)
        if F >= best_F:
            best_F = F
            torch.save(model.state_dict(), save_path)
            print("saved", save_path)

    print("Best match accuracy: %.2f %%" % best_F)


if __name__ == '__main__':
    main()
