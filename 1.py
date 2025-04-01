import sys
import os
import shutil
import torch
torch.manual_seed(42)  # 固定随机种子
"""
D:
cd D:\Paper\PaperCode4Paper3\Experiments\BDQ
tensorboard --logdir ./logs/BDQ

"""

# 将文件夹的路径添加到sys.path中
sys.path.append('D:\Paper\PaperCode4Paper3\Experiments\KAN_DQN')
import numpy as np
#%%
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from bdq_agent import BDQ_Agent
from DiscreteSwimmer import DiscreteSwimmerRLModel
logPath = "logs/BDQ"
if os.path.exists(logPath) and os.path.isdir(logPath):
    shutil.rmtree(logPath)
configs_dict = get_configs(file_dir="BDQ.yaml")
configs = argparse.Namespace(**configs_dict)
REGISTRY_ENV[configs.env_name] = DiscreteSwimmerRLModel
envs = make_envs(configs)
Agent = BDQ_Agent(configs, envs)
Agent.train(configs.running_steps // configs.parallels)
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
