import torch.nn as nn
import torch.nn.functional as F


class FCAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FCAgent, self).__init__()
        self.args = args

        self.model = nn.Sequential(
            nn.Linear(input_shape, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )

    def init_hidden(self):
        # make hidden states on same device as model
        # return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        raise NotImplementedError

    def forward(self, inputs, hidden_state):
        return self.model(inputs), None
        # x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # h = self.rnn(x, h_in)
        # q = self.fc2(h)
        # return q, h
