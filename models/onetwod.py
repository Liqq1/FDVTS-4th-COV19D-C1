
import torch
# from torchvision.models import resnet50
from timm.models import resnet50d
import torch.nn as nn
import torch.nn.functional as F


class OneTwoNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=4, use_con=True, num_stages=2, head='mlp', outdim=128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_con = use_con
        self.num_stages = num_stages
        ## stages: 1 or 2 or 3 or 4
        
        assert num_stages > 0 and num_stages < 5
        # self.preconv1 = nn.Conv2d(128, 3, 1, 1, 0)
        # if num_stages > 1:
        #     self.preconv2 = nn.Conv2d(64, 3, 1, 1, 0)
        # if num_stages > 2:
        #     self.preconv3 = nn.Conv2d(32, 3, 1, 1, 0)
        # if num_stages > 3:
        #     self.preconv4 = nn.Conv2d(16, 3, 1, 1, 0)
        self.preconv1 = nn.Conv2d(128, 3, 3, 1, 1)
        if num_stages > 1:
            self.preconv2 = nn.Conv2d(64, 3, 3, 1, 1)
        if num_stages > 2:
            self.preconv3 = nn.Conv2d(32, 3, 3, 1, 1)
        if num_stages > 3:
            self.preconv4 = nn.Conv2d(16, 3, 3, 1, 1)

        self.conv_stem = nn.Conv2d(3 * num_stages, 3, 3, 1, 1)
        
        self.relu = nn.ReLU(inplace=True)
        self.backbone = resnet50d(pretrained=pretrained)
        num_features = self.backbone.num_features
        
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        if self.use_con:
            if head == 'linear':
                self.mlp_head = nn.Linear(num_features, outdim)
            elif head == 'mlp':
                self.mlp_head = nn.Sequential(
                    nn.Linear(num_features, num_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(num_features, outdim)
                )
            else:
                raise NotImplementedError(
                    'head not supported: {}'.format(head))

        
        
    def forward(self, x):
        '''
        input:
            x: a list containing different slices 
            x[0]: [b, c, 128, h, w], suppose c=1 for now
            x[1]: [b, c, 64, h, w]
        '''
        # print(x[0].size())
        x[0] = self.relu(self.preconv1(x[0].squeeze(1)))
        if self.num_stages > 1:
            x[1] = self.relu(self.preconv2(x[1].squeeze(1)))
        if self.num_stages > 2:
            x[2] = self.relu(self.preconv3(x[2].squeeze(1)))
        if self.num_stages > 3:
            x[3] = self.relu(self.preconv4(x[3].squeeze(1)))
        
        # [b, 3, h, w] --> [b, 6, h, w] --> [b, 3, h, w]
        x = self.relu(self.conv_stem(torch.cat(x, dim=1)))
        # [b, chan, 7, 7]
        # feat = self.backbone.forward_features(x)
        # [b, chan]
        # feat = feat.flatten(2).mean(dim=-1)
        
        # out = self.backbone.fc(feat)
        # if self.use_con:
        #     feat = F.normalize(self.mlp_head(feat), dim = 1)
        #     return feat, out
        
        out = self.backbone(x)


        return x, out
            
if __name__ == '__main__':
    net = OneTwoNet(pretrained=True)
    ipt = torch.randn(2, 1, 128, 224, 224)
    ipt2 = torch.randn(2, 1, 64, 224, 224)
    ipt3 = torch.randn(2, 1, 32, 224, 224)
    ipt4 = torch.randn(2, 1, 16, 224, 224)
    feat, out = net([ipt, ipt2])
    feat, out = net([ipt, ipt2, ipt3, ipt4])
    print(feat.shape, out.shape)
