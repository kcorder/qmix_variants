from smac.env.multiagentenv import MultiAgentEnv
from pettingzoo.mpe import simple_spread_v2, simple_reference_v2
import numpy as np

MAPNAME_TO_ENV = {
    "simple_spread": simple_spread_v2.parallel_env,
    "simple_reference": simple_reference_v2.parallel_env,
}

class MPEEnv(MultiAgentEnv):

    def __init__(
            self,
            map_name="simple_spread",
            episode_limit=25,
            obs_last_action=False,
            seed=None,
            obs_instead_of_state=True,
            state_last_action=False,
    ):
        self.map_name = map_name
        self.episode_limit = episode_limit
        self.obs_instead_of_state = obs_instead_of_state

        self._mpe_env = MAPNAME_TO_ENV[map_name]()
        self.action_spaces = self._mpe_env.action_spaces
        # Discrete action spaces and homogeneous - just get first one
        self.n_actions = list(self.action_spaces.values())[0].n
        self.observation_spaces = self._mpe_env.observation_spaces
        self._max_obs_size = max(int(np.prod(space.shape))
                                 for space in self.observation_spaces.values())

        # self.n_agents = self._mpe_env.max_num_agents
        self.all_agents = self._mpe_env.possible_agents
        self.good_ags = [ag_name for ag_name in self.all_agents if "agent" in ag_name]
        self.adv_ags = [ag_name for ag_name in self.all_agents if "adversary" in ag_name]
        self.n_agents = len(self.good_ags)
        self.n_advs = len(self.adv_ags)

        self._obs = None

    def step(self, actions):
        action_dict = {self.all_agents[i]: int(actions[i]) for i in range(self.n_agents)}
        obss, rews, dones, infos = self._mpe_env.step(action_dict)
        self._obs = obss

        return sum(rews.values()), dones['agent_0'], infos['agent_0']

    def get_obs(self):
        """returns all agent observations in a list"""
        return [self.get_obs_agent(ag_id) for ag_id in self.all_agents]

    def pad_obs(self, obs, size):
        t = size - len(obs)
        return np.pad(obs, pad_width=(0, t), mode='constant')

    def get_obs_agent(self, agent_id):
        orig_obs = self._mpe_env.aec_env.observe(agent_id)
        return self.pad_obs(orig_obs, self._max_obs_size)

    def get_obs_size(self):
        return self._max_obs_size  # all will be padded - easier this way

    def get_state(self):
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat
        else:
            raise NotImplementedError

    def get_state_size(self):
        if self.obs_instead_of_state:
            return int(self.get_obs_size()) * len(self.all_agents)
        else:
            raise NotImplementedError

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(ag_id) for ag_id in self.all_agents]

    def get_avail_agent_actions(self, agent_id):
        aspace = self.action_spaces[agent_id]
        return [1] * aspace.n  # all actions are available always

    def get_total_actions(self):
        return self.n_actions

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "n_advs": self.n_advs,
                    "episode_limit": self.episode_limit}
        return env_info

    def reset(self):
        self._obs = self._mpe_env.reset()
        return self._obs

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

