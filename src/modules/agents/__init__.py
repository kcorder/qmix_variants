REGISTRY = {}

from .rnn_agent import RNNAgent
from .fc_agent import FCAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["fc"] = FCAgent