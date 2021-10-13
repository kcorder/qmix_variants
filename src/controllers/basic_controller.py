import copy

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn.functional as F
from itertools import chain

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        if args.action_selector is not None:
            self.action_selector = action_REGISTRY[args.action_selector](args)
        else:
            self.action_selector = None
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        if self.args.agent == 'rnn':
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape


class IndependentMAC:
    """MAC that doesn't use param sharing"""
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # agent outputs is now a list of each agent model's output tensors with shape (batch, num_actions)
        chosen_actions = []
        # for ag_i, output_i in enumerate(agent_outputs):
        for ag_i in range(self.n_agents):
            action_i = self.action_selector.select_action(agent_outputs[bs, ag_i], avail_actions[bs, ag_i],
                                                          t_env, test_mode=test_mode)
            chosen_actions.append(action_i)
        chosen_actions = th.cat(chosen_actions)
        # chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agents_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, state_outs = [], []
        for ag_i, input_i in enumerate(agents_inputs):
            h_in = self.hidden_states[ag_i] if self.hidden_states else None
            a_out, h_out = self.agents[ag_i](input_i, h_in)
            agent_outs.append(a_out)
            state_outs.append(h_out)
            # agent_outs, self.hidden_states = self.agent(agents_inputs, self.hidden_states)
        self.hidden_states = state_outs if self.hidden_states else None

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0


        return th.stack(agent_outs, dim=1)


    def init_hidden(self, batch_size):
        if self.args.agent == "rnn":
            self.hidden_states = [ag.init_hidden().unsqueeze(0).expand(batch_size, 1, -1) for ag in self.agents]
        else:
            self.hidden_states = None

    def parameters(self):
        return chain.from_iterable(agent.parameters() for agent in self.agents)

    def load_state(self, other_mac):
        for ag_i in range(self.n_agents):
            agent, other = self.agents[ag_i], other_mac.agents[ag_i]
            agent.load_state_dict(other.state_dict())

    def cuda(self):
        for ag_i in range(self.n_agents):
            self.agents[ag_i].cuda()

    def save_models(self, path):
        for ag_i in range(self.n_agents):
            th.save(self.agents[ag_i].state_dict(), f"{path}/agent{ag_i}.th")

    def load_models(self, path):
        for ag_i in range(self.n_agents):
            self.agents[ag_i].load_state_dict(th.load(f"{path}/agent{ag_i}.th", map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agents = [agent_REGISTRY[self.args.agent](input_shape, self.args)
                       for _ in range(self.args.n_agents)]

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        num_agents = self.n_agents

        # Splitting the batch per agent:
        # agent_inputs is num_agents length list of lists (one for each agent inputs)
        ag_inputs = [ [batch['obs'][:, t, ag_i:ag_i+1]] for ag_i in range(self.n_agents)]

        if self.args.obs_last_action:
            if t == 0:
                for ag_i in range(num_agents):
                    ag_inputs[ag_i].append(th.zeros_like(batch["actions_onehot"][:, t, ag_i:ag_i+1]))
            else:
                for ag_i in range(num_agents):
                    ag_inputs[ag_i].append(batch["actions_onehot"][:, t - 1, ag_i:ag_i+1])
        if self.args.obs_agent_id:
            # inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            for ag_i in range(num_agents):
                onehot = F.one_hot(th.tensor([0], device=batch.device), num_classes=num_agents)
                ag_inputs[ag_i].append(onehot.unsqueeze(0).expand(bs, -1, -1))

        # inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        # _ag_inputs = copy.deepcopy(ag_inputs)
        for ag_i in range(num_agents):
            ag_inputs[ag_i] = th.cat([inputs.reshape(bs, -1) for inputs in ag_inputs[ag_i]], dim=1)
        # inputs = th.cat([data.reshape(bs*self.n_agents, -1) for data in inputs_i for inputs_i in ag_inputs], dim=1)
        return ag_inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape