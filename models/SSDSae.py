import torch.nn as nn

Outputsize = 100
inputflag = 0

# -------------------inputsize == 1024
class encoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(encoder, self).__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True))
        self.fc5 = nn.Sequential(
            nn.Linear(100, Outputsize), )

    def forward(self, x):
        global inputflag
        if x.shape[2] == 512:
            inputflag = 0
            out = x.view(x.size(0), -1)
            out = self.fc1(out)
        else:
            inputflag = 1
            out = x.view(x.size(0), -1)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        return out


class decoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(decoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(Outputsize, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(100, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(
            nn.Linear(200, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(1024, 512), )

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        if inputflag == 0:
            out = self.fc4(out)
            out = self.fc5(out)
        else:
            out = self.fc4(out)

        return out


class classifier(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(classifier, self).__init__()
        self.fc1 = nn.Sequential(
            # 这一个ReLU是否需要加，并不是很确定
            # inplace = True 能够提高程序运行的效率
            nn.ReLU(inplace=True),
            nn.Linear(Outputsize, 1024),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 100),
            nn.ReLU(inplace=True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 10),
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        label = self.fc3(out)
        return label
