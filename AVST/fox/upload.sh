#!/bin/bash
# Copy code and data from this machine to Fox. Run locally (Git Bash) from the repo root:
#     bash AVST/fox/upload.sh code     # code, scripts, pretrained weights (small; re-run after code changes)
#     bash AVST/fox/upload.sh data     # json + vggish + res18_14x14 features + grounding/AVST checkpoints (~8 GB)
#     bash AVST/fox/upload.sh video    # raw videos for DINOv2 frame extraction later (~7 GB)
# Streams a tar archive over one ssh connection, so you only do the 2FA login once per call.
set -euo pipefail

FOX_USER=${FOX_USER:-ec-williaan}
FOX=${FOX:-$FOX_USER@fox.educloud.no}
DEST=${DEST:-/fp/projects01/ec12/$FOX_USER/MusiQAl}

case "${1:-}" in
  code)
    FILES=(AVST/net_grd_avst/*.py AVST/grounding_gen/*.py AVST/feat_script AVST/tools AVST/fox
           AVST/pretrained/resnet18-5c106cde.pth AVST/requirements_final.txt .gitattributes .gitignore)
    ;;
  data)
    FILES=(AVST/data/json AVST/data/ans_vocab.txt AVST/data/ques_vocab.txt
           AVST/data/feats/vggish AVST/data/feats/res18_14x14
           AVST/grounding_gen/models_grounding_gen/main_grounding_gen_best.pt
           AVST/net_grd_avst/avst_models/avst_ours.pt AVST/net_grd_avst/avst_models/avst_lr.pt)
    ;;
  video)
    FILES=(AVST/data/video)
    ;;
  *)
    echo "usage: bash AVST/fox/upload.sh code|data|video"; exit 1
    ;;
esac

echo "Uploading to $FOX:$DEST"
du -shc "${FILES[@]}" | tail -1
tar --exclude='__pycache__' -cf - "${FILES[@]}" | ssh "$FOX" "mkdir -p '$DEST' && tar -xf - -C '$DEST'"
echo "done"
