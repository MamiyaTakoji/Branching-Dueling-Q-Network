import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class BDQ_Learner(Learner):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module):
        super(BDQ_Learner, self).__init__(config, policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1.0,
                                                           end_factor=self.end_factor_lr_decay,
                                                           total_iters=self.config.running_steps)
        self.gamma = config.gamma
        self.sync_frequency = config.sync_frequency
        self.mse_loss = nn.MSELoss()
        self.one_hot = nn.functional.one_hot
        self.n_actions = self.policy.action_dim
    def update(self, **samples):
        self.iterations += 1
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)
        
        _, evalAs1, evalQs1 = self.policy(obs_batch)
        _, targetAs, targetQs = self.policy.target(next_batch)
        _, evalAs2, evalQs2 = self.policy(next_batch)
        num_action_streams = len(evalQs2)
        targetQs_ = []
        rows = torch.arange(targetQs[0].size(0), device=targetQs[0].device)
        #获取目标Q值
        for i in range(num_action_streams):
            #每一个分量选取evalAs的分量作为Q值然后求和
            targetQ = targetQs[i][rows,evalAs2[i]]
            targetQs_.append(targetQ)
        targetQ_ = sum(targetQs_)/num_action_streams
        targetQ_ = rew_batch + self.gamma*(1 - ter_batch)*targetQ_
        #获取实际（？）Q值
        targetQs_ = [targetQ_ ]*num_action_streams
        preditQs = []
        for i in range(num_action_streams):
            preditQ = evalQs1[i][rows,act_batch[:,i].to(torch.int32)]
            preditQs.append(preditQ)
        loss = 0
        for i in range(num_action_streams):
            loss += self.mse_loss(preditQs[i],targetQs_[i])
        loss = loss/num_action_streams
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # hard update for target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        if self.distributed_training:
            info = {
                f"Qloss/rank_{self.rank}": loss.item(),
                f"learning_rate/rank_{self.rank}": lr,
                #f"predictQ/rank_{self.rank}": predictQ.mean().item()
            }
        else:
            info = {
                "Qloss": loss.item(),
                "learning_rate": lr,
                #"predictQ": predictQ.mean().item()
            }

        return info
        
        #计算误差
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        