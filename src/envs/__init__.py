from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
from gym.spaces import Discrete

def env_fn(env, **kwargs) -> MultiAgentEnv:
    # Preprocess kwargs
    pass
    # Make env
    env_obj = env(**kwargs)
    # Postprocess env
    env_obj.action_space = Discrete(env_obj.n_actions)

    return env_obj

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
