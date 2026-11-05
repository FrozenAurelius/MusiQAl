# MusiQAl: Music Question-Answering through Audio-Video Fusion

# Description

The MusiQAl Dataset is a comprehensive collection designed to advance the field of music question-answering (MQA). It features 292 carefully selected videos and 11,793 question-answer pairs, totaling five hours of music performance footage. Performances are sourced from notable projects and public datasets, along with a diverse range of YouTube videos.

## Content and Representation
47 Music Instruments: Spanning a wide range of musical traditions and genres.
Performances Across 18 Countries and Five Continents: Showcasing a global perspective on music.
11 Question Types: Covering audio, visual, and audio-visual scenarios.
The dataset highlights four specific aspects of music performance: instrument playing, dancing, singing, and their combinations. Recognizing the cultural significance of rhythm and dance, MusiQAl includes music made for dance, integral to many traditions.

## Data Availability
For access to the dataset, please visit this link: 10.5281/zenodo.13623448

# Requirements

```bash 
pip install -r requirements.txt
```

# Usage

## Clone the repository

``` bash
git clone https://github.com/MuTecEn/MusiQAl.git
```

## Download data

Please make sure to download the wav files, video frames, features, and JSON files from this link: https://zenodo.org/records/13623449

## Run AVST

```bash
python ${DIR}/net_grd_avst/main_avst.py --label_train ${DIR}/data/json/avqa-train.json --label_val ${DIR}/data/json/avqa-val.json --label_test ${DIR}/data/json/avqa-test.json --batch-size 16 --epochs 30 --mode train --model_save_dir ${DIR}/net_grd_avst/avst_models/
```

## Run LAVISH 

```bash
python ${DIR}/net_grd_avst/main_avst.py --Adapter_downsample=8 --audio_dir=${DIR}/data/feats/vggish --epochs=30 -- is_before_layernorm=1 --is_bn=0 --is_gate=1 --is_multimodal=1 --is_post_layernorm=1 --is_vit_ln=1 --num_conv_group=4 --num_tokens=64 --num_workers=16 --video_res14x14_dir=${DIR}/data/feats/res18_14x14 --video_dir=${DIR}/data/frames --wandb=1 --mode train
```

## Test and Compare Results

AVST:

```bash
python ${DIR}/net_grd_avst/main_avst.py --label_train ${DIR}/data/json/avqa-train.json --label_val ${DIR}/data/json/avqa-val.json --label_test ${DIR}/data/json/avqa-test.json --batch-size 16 --epochs 30 --mode test --model_save_dir ${DIR}/net_grd_avst/avst_models/
```

LAVISH: 
```bash
python ${DIR}/net_grd_avst/main_avst.py --Adapter_downsample=8 --audio_dir=${DIR}/data/feats/vggish --epochs=30 -- is_before_layernorm=1 --is_bn=0 --is_gate=1 --is_multimodal=1 --is_post_layernorm=1 --is_vit_ln=1 --num_conv_group=4 --num_tokens=64 --num_workers=16 --video_res14x14_dir=${DIR}/data/feats/res18_14x14 --video_dir=${DIR}/data/frames --wandb=1 --mode test
```
## Results Table

![image](https://github.com/user-attachments/assets/6d38831d-67ef-45e2-918c-842d616e46a3)

## DINOv2 experiments (branch `grounding-dinov2`)

Code for the paper *Can Improving AI Models' Eyes Lead to Better Music Perception?*, which replaces AVST's ResNet-18 features with DINOv2 (ViT-B/14 with registers). All jobs run from the repository root on the Fox cluster (Slurm).

| Paper | Script | Notes |
|---|---|---|
| Feature extraction | `AVST/feat_script/extract_visual_feat_14x14/extract_dinov2_feat.py` | 60 frames per video, patch tokens only, saved as `[60, 768, 16, 16]` |
| Table I (grounding stage) | `AVST/Grounding.sh` → `AVST/grounding_gen/main_grd_gen_feat.py` | re-trains the grounding stage on saved features |
| Table II (setups A–D, 3 seeds) | `AVST/Test.sh` | settings via `sbatch --export`: `FEATS`, `VDIM`, `NAME`, `GRD`, `TLAYERS`, `SEED` |
| Table III, Figs. 2–3 (visual features) | `AVST/FeatureViz.sh` → `AVST/tools/feature_viz.py` | nearest-neighbour test, PCA plots, patch maps, similarity over time |
| Table IV (input ablations) | `AVST/Ablation.sh` → `--ablate` in `AVST/net_grd_avst/main_avst.py` | audio, video or both replaced by zeros, noise or another video's input |
| Fig. 4 (sound localisation maps) | `AVST/AttentionMaps.sh` → `AVST/tools/attention_maps.py` | seed-1 models, eight test videos |
| Mean ± std tables | `AVST/tools/summarize_results.py avst-*.out ablate-*.out` | reads the Slurm logs |

The model changes are in `AVST/net_grd_avst/net_avst.py` (`--visual_dim` for the 768→512 layer, `--temporal_layers` for the temporal Transformer).
