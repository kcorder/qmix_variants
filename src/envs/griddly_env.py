from smac.env.multiagentenv import MultiAgentEnv
import griddly

class GriddlyEnv(MultiAgentEnv):

    def __init__(
            self,
            map_name="mirror",
            episode_limit=25,
            obs_last_action=False,
            seed=None,
    ):
        self.map_name = map_name

        self._mpe_env = None

        self.n_agents = self._mpe_env.n_agents 