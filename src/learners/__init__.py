from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .cq_learner import CQLearner
from .facmaddpg_learner import FacMADDPGLearner
from .maddpg_learner import MADDPGLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["cq_learner"] = CQLearner
REGISTRY["facmaddpg_learner"] = FacMADDPGLearner
REGISTRY["maddpg_learner"] = MADDPGLearner