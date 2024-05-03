import torch
import torch.nn as nn
from torch.nn import functional as F


class SemanticEmbedding(nn.Module):
    def __init__(self, args, mesh_dim=71, report_dim=761, embed_size=512):
        super(SemanticEmbedding, self).__init__()
        self.mesh_tf = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size // 2, embed_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size // 4, mesh_dim)
        )

        self.report_tf = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size // 2, embed_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size // 4, report_dim)
        )
        self.bn = nn.BatchNorm1d(num_features=embed_size, momentum=0.1)
        self.w1 = nn.Linear(in_features=mesh_dim + report_dim, out_features=embed_size)
        self.w2 = nn.Linear(in_features=embed_size, out_features=embed_size)
        self.relu = nn.ReLU()
        self.logit = nn.Linear(60, 31)
        self.dropout = nn.Dropout(0.2)
        self.__init_weight()
        self.target_dim = 60
        self.sigm = nn.Sigmoid()

    def __init_weight(self):
        self.w1.weight.data.uniform_(-0.1, 0.1)
        self.w1.bias.data.fill_(0)
        self.w2.weight.data.uniform_(-0.1, 0.1)
        self.w2.bias.data.fill_(0)

    def forward(self, avg, pred_output):
        avg_visual = avg.unsqueeze(1)
        pred_output2 = F.pad(pred_output, (0, 0, 0, self.target_dim - pred_output.shape[1]), 'constant', 0)
        pred = pred_output2.permute(0, 2, 1)
        visual_text = torch.matmul(avg_visual, pred).squeeze(1)
        outputs = self.sigm(self.logit(visual_text))
        return outputs


class classfication(nn.Module):
    def __init__(self, distiller_num, avg_dim=1024):
        super(classfication, self).__init__()
        self.logit = nn.Linear(avg_dim, distiller_num)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, avg):
        avg_visual = self.dropout(avg)
        x = self.logit(avg_visual)
        outputs = self.sigm(x)
        return outputs
