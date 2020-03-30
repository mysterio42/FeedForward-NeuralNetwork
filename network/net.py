import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)

        self.tanh = nn.Tanh()

        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        out = self.fc1(x)

        out = self.tanh(out)

        out = self.fc2(out)

        return out
