import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.cov1x1 = nn.Conv2d(in_channels=2048, out_channels=args.nhidden, kernel_size=(1, 1))
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained

        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        if self.pretrained is True: print('first init the imagenet pretrained!')

    def forward(self, images):

        patch_feats = self.model(images)

        att_feat_it = self.cov1x1(patch_feats)
        avg_feat_it = self.avg_fnt(att_feat_it).squeeze().reshape(-1, att_feat_it.size(1))

        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats, att_feat_it, avg_feat_it


