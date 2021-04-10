import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class FClayer(nn.Module):  # define fully connected layer with Leaky ReLU activation function
    def __init__(self, innodes, nodes):
        super(FClayer, self).__init__()
        self.fc = nn.Linear(innodes, nodes)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.fc(x)
        out = self.act(out)
        return out


# define custom model named wave net, which was coined after seeing the nodes sway
class WaveNET(nn.Module):
    def __init__(self, block, planes, nodes, num_classes=3):
        super(WaveNET, self).__init__()
        self.innodes = 5

        self.layer1 = self.make_layer(block, planes[0], nodes[0])
        self.layer2 = self.make_layer(block, planes[1], nodes[1])
        self.layer3 = self.make_layer(block, planes[2], nodes[2])

        self.fin_fc = nn.Linear(self.innodes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def make_layer(self, block, planes, nodes):

        layers = []
        layers.append(block(self.innodes, nodes))
        self.innodes = nodes
        for _ in range(1, planes):
            layers.append(block(self.innodes, nodes))

        return nn.Sequential(*layers)

    def forward_impl(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fin_fc(x)

        return x

    def forward(self, x):
        return self.forward_impl(x)
