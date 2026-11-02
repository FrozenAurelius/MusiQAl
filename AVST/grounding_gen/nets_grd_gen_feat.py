import torch
import torch.nn as nn
import torch.nn.functional as F


class AVQA_AVatt_Grounding_Feat(nn.Module):
    # same as AVQA_AVatt_Grounding, but on saved features instead of frames + resnet18.
    # layer names match AVQA_Fusion_Net, so the weights can be loaded into AVST

    def __init__(self, visual_dim=512):
        super(AVQA_AVatt_Grounding_Feat, self).__init__()

        # dinov2: 768 -> 512
        self.fc_visual = nn.Linear(visual_dim, 512) if visual_dim != 512 else None

        # for features
        self.fc_a1 =  nn.Linear(128, 512)
        self.fc_a2=nn.Linear(512,512)

        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_gl=nn.Linear(1024,512)
        self.tanh = nn.Tanh()


    def forward(self, audio, visual):
        '''
            input audio shape:  [B, 2, 128]        (same second twice)
            input visual shape: [B, 2, C, H, W]    (matching frame, frame from another video)
        '''

        ## audio features
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat=self.fc_a2(audio_feat)                   # [B, 2, 512]
        (B, T, C) = audio_feat.size()
        audio_feat = audio_feat.view(B*T, C)                # [B*2, 512]

        ## dinov2 feats [B, 2, 768, H, W] -> [B, 2, 512, H, W]
        if self.fc_visual is not None:
            visual = self.fc_visual(visual.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3).contiguous()

        (B, T, C, H, W) = visual.size()
        v_feat_out = visual.view(B * T, C, H, W)            # [B*2, 512, H, W]
        v_feat=self.avgpool(v_feat_out)
        visual_feat_before_grounding=v_feat.squeeze(-1).squeeze(-1)     # [B*2, 512]

        v_feat = v_feat_out.view(B * T, C, H * W)
        v_feat = v_feat.permute(0, 2, 1)  # B, HxW, C
        visual = nn.functional.normalize(v_feat, dim=2)

        ## audio-visual grounding
        audio_feat_aa = audio_feat.unsqueeze(-1)            # [B*2, 512, 1]
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)
        visual_feat = visual
        x2_va = torch.matmul(visual_feat, audio_feat_aa).squeeze(-1)

        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)       # [B*2, 1, HxW]
        visual_feat_grd = torch.matmul(x2_p, visual_feat)
        visual_feat_grd = visual_feat_grd.squeeze(-2)       # [B*2, 512]

        visual_gl=torch.cat((visual_feat_before_grounding,visual_feat_grd),dim=-1)
        visual_feat_grd=self.tanh(visual_gl)
        visual_feat_grd=self.fc_gl(visual_feat_grd)

        # combine a and v
        feat = torch.cat((audio_feat, visual_feat_grd), dim=-1)     # [B*2, 1024]

        feat = F.relu(self.fc1(feat))   # (1024, 512)
        feat = F.relu(self.fc2(feat))   # (512, 256)
        feat = F.relu(self.fc3(feat))   # (256, 128)
        feat = self.fc4(feat)   # (128, 2)

        return  feat
