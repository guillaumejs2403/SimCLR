import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import models.decoder as decoder
import models.resnet_cifar as cresnet

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False),
                            "resnet18-cifar": cresnet.ResNet18(),
                            "resnet50-cifar": cresnet.ResNet50()}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)

        # import pdb; pdb.set_trace()
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x


class ResNetSimCLR_AE(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_AE, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False),
                            "resnet18-cifar": cresnet.ResNet18(),
                            "resnet50-cifar": cresnet.ResNet50()}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        self.decoder = decoder.Decoder(num_ftrs)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)

        d = self.decoder(h)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x, d 
