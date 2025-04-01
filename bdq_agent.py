import torch
from argparse import Namespace
from xuance.common import Union
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.agents import OffPolicyAgent
from bdq_learner import BDQ_Learner
from deterministic import BDQPolicy
import numpy as np
from tqdm import tqdm
from copy import deepcopy



class BDQ_Agent(OffPolicyAgent):
    def _build_learner(self, config, policy):
        return BDQ_Learner(config,policy)
    def exploration(self, pi_actions):
        """Returns the actions for exploration.

        Parameters:
            pi_actions: The original output actions.

        Returns:
            explore_actions: The actions with noisy values.
        """
        explore_actions_numpy = []
        explore_actions = pi_actions
        for i in range(len(explore_actions)):
            explore_action_numpy = explore_actions[i].detach().cpu().numpy()
            random_action = np.random.choice(self.action_space.nvec[i],self.n_envs)
            mask = np.random.rand(self.n_envs)<self.e_greedy
            explore_action_numpy[mask] = random_action[mask] 
            explore_actions_numpy.append(explore_action_numpy)
        return explore_actions_numpy
            
        # if self.e_greedy is not None:
        #     random_actions = np.random.choice(self.action_space.n, self.n_envs)
        #     if np.random.rand() < self.e_greedy:
        #         explore_actions = random_actions
        #     else:
        #         explore_actions = pi_actions.detach().cpu().numpy()
        # elif self.noise_scale is not None:
        #     explore_actions = pi_actions + np.random.normal(size=pi_actions.shape) * self.noise_scale
        #     explore_actions = np.clip(explore_actions, self.actions_low, self.actions_high)
        # else:
        #     explore_actions = pi_actions.detach().cpu().numpy()
        return explore_actions
    def train(self, train_steps):
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, test_mode=False)
            acts = policy_out['actions']
            acts = np.array(acts)
            acts = list(acts.T)
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)

            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))
            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                train_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(train_info, self.current_step)

            self.returns = self.gamma * self.returns + rewards
            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
                        self.ret_rms.update(self.returns[i:i + 1])
                        self.returns[i] = 0.0
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info[f"Episode-Steps/rank_{self.rank}/env-{i}"] = infos[i]["episode_step"]
                            step_info[f"Train-Episode-Rewards/rank_{self.rank}/env-{i}"] = infos[i]["episode_score"]
                        else:
                            step_info[f"Episode-Steps/rank_{self.rank}"] = {f"env-{i}": infos[i]["episode_step"]}
                            step_info[f"Train-Episode-Rewards/rank_{self.rank}"] = {
                                f"env-{i}": infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)

            self.current_step += self.n_envs
            self._update_explore_factor()
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(BDQ_Agent, self).__init__(config, envs)
        self.start_greedy, self.end_greedy = config.start_greedy, config.end_greedy
        self.e_greedy = config.start_greedy
        #self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy / self.n_envs)
        self.delta_egreedy = (self.start_greedy - self.end_greedy) / (config.decay_step_greedy)
        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()  # build memory
        self.learner = self._build_learner(self.config, self.policy)  # build learner
    
    def _build_policy(self) -> Module:
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representation.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy.
        if self.config.policy == "BDQ_Policy":
            policy = BDQPolicy(action_space = self.action_space,
                               representation = representation,
                               actionValueNet_hidden_sizes = self.config.actionValueNet_hidden_sizes,
                               stateValueNet_hidden_sizes = self.config.stateValueNet_hidden_sizes,
                               normalize = normalize_fn,
                               initialize = initializer,
                               activation = activation,
                               device = device)
        else:
            raise AttributeError(f"{self.config.agent} does not support the policy named {self.config.policy}.")

        return policy
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    