import torch
import numpy as np
from xuance.common import Sequence, Optional, Callable, Union, Dict
from copy import deepcopy
from gym.spaces import Space, Discrete, MultiDiscrete
from xuance.torch import Module, Tensor, DistributedDataParallel
from xuance.torch.utils import ModuleType
from core import BDQhead

class BDQPolicy(Module):
    def __init__(self,
                 action_space:MultiDiscrete,
                 representation: Module,
                 actionValueNet_hidden_sizes: Sequence[int],
                 stateValueNet_hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(BDQPolicy,self).__init__()
        #初始化BDQhead
        self.total_actions = action_space.nvec[0]*len(action_space.nvec)
        self.num_branches = len(action_space.nvec)
        self.action_dim = action_space.nvec
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.target_representation = deepcopy(representation)
        self.eval_BDQhead = BDQhead(
            state_dim = self.representation.output_shapes['state'][0],
            actionValueNet_hidden_sizes = actionValueNet_hidden_sizes,
            stateValueNet_hidden_sizes = stateValueNet_hidden_sizes,
            total_actions = self.total_actions,
            num_branches = self.num_branches,
            normalize = normalize,
            initialize = initialize,
            activation = activation,
            device = device
            )
        self.target_BDQhead = deepcopy(self.eval_BDQhead)
    def forward(self, observation: Union[np.ndarray, dict]):
        outputs = self.representation(observation)
        evalQ = self.eval_BDQhead(outputs['state'])
        argmax_actions = []
        #evalQ是一个Tensor的list，list的长度为num_branches，
        # 其中的每一个元素代表那个动作的每一个分动作的价值
        #需要返回 outputs, 最优动作, evalQ
        for q in evalQ:
            argmax_action = q.argmax(axis = 1)
            argmax_actions.append(argmax_action)
        return outputs, argmax_actions, evalQ
    def target(self, observation: Union[np.ndarray, dict]):
        outputs_target = self.target_representation(observation)
        targetQ = self.target_BDQhead(outputs_target['state'])
        argmax_actions = []
        #evalQ是一个Tensor的list，list的长度为num_branches，
        # 其中的每一个元素代表那个动作的每一个分动作的价值
        #需要返回 outputs, 最优动作, evalQ
        for q in targetQ:
            argmax_action = q.argmax(axis = 1)
            argmax_actions.append(argmax_action)
            argmax_actions_numpy = []
            for argmax_action in argmax_actions:
                argmax_actions_numpy.append(argmax_action.detach())
            targetQ_numpy = []
            for _targetQ in targetQ:
                targetQ_numpy.append(_targetQ.detach())
        return outputs_target, argmax_actions_numpy, targetQ_numpy
        # elif target_version == "mean":
        #     for dim in range(num_action_streams):
        #         selected_a = tf.argmax(selection_q_tp1[dim], axis=1)
        #         selected_q = tf.reduce_sum(tf.one_hot(selected_a, num_actions_pad) * q_tp1[dim], axis=1) 
        #         masked_selected_q = (1.0 - done_mask_ph) * selected_q
        #         if dim == 0:
        #             mean_next_q_values = masked_selected_q
        #         else:
        #             mean_next_q_values += masked_selected_q 
        #     mean_next_q_values /= num_action_streams
        #     target_q_values = [rew_t_ph + gamma * mean_next_q_values] * num_action_streams
        #TensorFlow代码
        #target_q_vakues见公式6
    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_BDQhead.parameters(), self.target_BDQhead.parameters()):
            tp.data.copy_(ep)
        
    
def Test():
    state_dim = 4;action_dim = 5
    blockType = ['K','K','K','M']
    kanConfig = {
        0:{"grid_size":5, 'spline_order':3},
        1:{"grid_size":5, 'spline_order':3},
        2:{"grid_size":5, 'spline_order':3},
        3:{"grid_size":5, 'spline_order':3},
        }
    hidden_sizes = [6,7,8]
    from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
    activationMLP = ActivationFunctions["relu"]
    activationKAN = ActivationFunctions['relu']
    device = "cuda:0"
    representation = BasicHyperQhead(state_dim = state_dim,
                        n_actions = action_dim, 
                        blockType = blockType, 
                        kanConfig = kanConfig, 
                        hidden_sizes = hidden_sizes,
                        activationMLP = activationMLP,
                        activationKAN = activationKAN,
                        device = device)
    from gym.spaces import Box,Discrete
    T = BasicHyperQnetwork(
        action_space = Discrete(4),
        representation = representation,
        kanConfig = kanConfig,
        blockType = blockType,
        hidden_size = hidden_sizes,
        activationMLP = activationMLP,
        activationKAN = activationKAN,
        device = device
        )
if __name__ == "__main__":
    Test()































