from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .cq_learner import CQLearner
from .maddpg_learner import MADDPGLearner
from .facmac_learner import FACMACLearner
from .facmac_learner_discrete import FACMACDiscreteLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["cq_learner"] = CQLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["facmac_learner"] = FACMACLearner
REGISTRY["facmac_learner_discrete"] = FACMACDiscreteLearner