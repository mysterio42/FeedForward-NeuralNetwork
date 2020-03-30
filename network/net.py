import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.tanh1 = nn.Tanh()

        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.tanh3 = nn.Tanh()

        self.fc4 = nn.Linear(in_features=hidden_dim,out_features=out_dim)

    def hidden_1_layer(self, x):
        return self.tanh1(self.fc1(x))

    def hidden_2_layer(self, x):
        return self.tanh2(self.fc2(x))

    def hidden_3_layer(self,x):
        return self.tanh3(self.fc3(x))

    def readout_layer(self, x):
        return self.fc4(x)

    def forward(self, x):
        return self.readout_layer(self.hidden_3_layer(self.hidden_2_layer(self.hidden_1_layer(x))))
