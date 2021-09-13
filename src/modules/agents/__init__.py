REGISTRY = {}

from .rnn_agent import RNNAgent
from .fc_agent import FCAgent
from .comix_agent import CEMAgent, NAFAgent
from .mlp_agent import MLPAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["fc"] = FCAgent
REGISTRY["naf"] = NAFAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["mlp"] = MLPAgent