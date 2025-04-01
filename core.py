import torch
import torch.nn as nn
from xuance.common import Sequence, Optional, Callable, Union, Dict
from xuance.torch import Tensor, Module
from xuance.torch.utils import ModuleType, mlp_block
class BDQhead(Module):
    #num_action_branches == 1的时候效果似乎等于DuelDQN，至少要对吧
    #虽然网络的结构很复杂但好像确实只需要一个网络就够了
    def __init__(self,
                 state_dim:int,
                 actionValueNet_hidden_sizes: Sequence[int],
                 stateValueNet_hidden_sizes: Sequence[int],
                 total_actions:int,
                 num_branches:int,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 aggregator = 'reduceLocalMean',):
        super(BDQhead,self).__init__()
        assert num_branches is not None, "必须指定动作分支数量"
        assert total_actions is not None, "必须指定总动作数量"
        assert state_dim is not None, "必须指定输入维度"
        self.state_dim = state_dim
        self.aggregator = aggregator
        self.num_branches = num_branches
        self.actionValueNet = self.ActionValueNet(
         state_dim, 
         actionValueNet_hidden_sizes, 
         total_actions, 
         num_branches,
         normalize,
         initialize,
         activation,
         device)
        self.stateValueNet = self.StateValueNet(
         state_dim, 
         stateValueNet_hidden_sizes, 
         normalize,
         initialize,
         activation,
         device)
    def _dueling_aggregation(self, action_scores, state_values):
        # 根据聚合方法处理优势值
        if self.aggregator == 'reduceLocalMean':
            adjusted_actions = [a - a.mean(dim=1, keepdim=True) for a in action_scores]
        elif self.aggregator == 'reduceGlobalMean':
            global_mean = torch.stack(action_scores).mean(dim=0)
            adjusted_actions = [a - global_mean for a in action_scores]
        elif self.aggregator == 'reduceLocalMax':
            adjusted_actions = [a - a.max(dim=1, keepdim=True)[0] for a in action_scores]
        else:  # naive
            adjusted_actions = action_scores
        
        # 组合状态值和优势值
        q_values = []
        for i in range(self.num_branches):
            q_values.append(state_values + adjusted_actions[i])
        return q_values
    def forward(self,x):
        action_scores = self.actionValueNet(x)
        state_values = self.stateValueNet(x)
        # 返回一个Tensor的list，list的长度为num_branches，
        # 其中的每一个元素代表那个动作的每一个分动作的价值
        return self._dueling_aggregation(action_scores, state_values)
    class ActionValueNet(Module):
        def __init__(self, 
                     state_dim, 
                     hidden_sizes, 
                     total_actions, 
                     num_branches,
                     normalize: Optional[ModuleType] = None,
                     initialize: Optional[Callable[..., Tensor]] = None,
                     activation: Optional[ModuleType] = None,
                     device: Optional[Union[str, int, torch.device]] = None
                     ):
            super(BDQhead.ActionValueNet, self).__init__()
            self.num_branches = num_branches
            self.actions_per_branch = total_actions // num_branches
            #按道理要实现distributed_single_stream的情况，但是之后再说吧
            self.branches = nn.ModuleList()
            for _ in range(num_branches):
                branch_layers = []; input_shape = (state_dim,)
                for h in hidden_sizes:
                    mlp, input_shape = mlp_block(input_shape[0], h,  normalize, activation, initialize, device)
                    branch_layers.extend(mlp)
                branch_layers.extend(mlp_block(input_shape[0], self.actions_per_branch, None, None, None, device)[0])
                self.branches.append(nn.Sequential(*branch_layers))
        def forward(self,x):
            return [branch(x) for branch in self.branches]
    #只实现not independent的情况
    class StateValueNet(Module):
        def __init__(self,state_dim, hidden_sizes,
                      normalize: Optional[ModuleType] = None,
                      initialize: Optional[Callable[..., Tensor]] = None,
                      activation: Optional[ModuleType] = None,
                      device: Optional[Union[str, int, torch.device]] = None):
            super().__init__()
            layers = []
            input_shape = (state_dim,)
            for h in hidden_sizes:
                mlp, input_shape = mlp_block(input_shape[0], h,  normalize, activation, initialize, device)
                layers.extend(mlp) 
            layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return self.model(x)