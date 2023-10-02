import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random
from copy import deepcopy
from config_reader import MODEL_SIZE
 
# ------------------------------------- #
# 经验回放池
# ------------------------------------- #
 
class ReplayBuffer:
    def __init__(self, capacity):  # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = collections.deque(maxlen=capacity)
    # 在队列中添加数据
    def add(self, state, action, reward, next_state):
        # 以list类型保存
        self.buffer.append([state, action, reward, next_state])
    # 在队列中随机取样batch_size组数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state = zip(*transitions)
        for index,tensor in enumerate(state):
            if index == 0:
                state_tensor = deepcopy(tensor)
            else:
                state_tensor = torch.cat([state_tensor,tensor],dim = 0)
        for index,tensor in enumerate(next_state):
            if index == 0:
                next_state_tensor = deepcopy(tensor)
            else:
                next_state_tensor = torch.cat([next_state_tensor,tensor],dim = 0)
        return state_tensor, action, reward, next_state_tensor
    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)
 
# ------------------------------------- #
# 策略网络
# ------------------------------------- #
 
class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions,n_layers):
        super(PolicyNet, self).__init__()
        # 环境可以接受的动作最大值
        # 只包含一个隐含层
        self.fc1 = nn.Linear(n_states, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, n_hiddens)
        self.fc4 = nn.Linear(n_hiddens, n_layers)
        # self.action_bound = np.array()
        # self.fc1 = nn.Linear(n_states, n_hiddens)
        # self.fc2 = nn.Linear(n_hiddens, n_layers)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        x = F.relu(x)
        x = self.fc3(x)  # [b,n_hiddens]-->[b,n_actions]
        x = F.relu(x)
        x = self.fc4(x)  # [b,n_hiddens]-->[b,n_actions]
        x = F.tanh(x)  # 将数值调整到 [-1,1]
        # x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        # x = F.relu(x)
        # x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        # x= torch.tanh(x)  # 将数值调整到 [-1,1]
        # x = x * self.action_bound  # 缩放到 [-action_bound, action_bound]
        return x
 
# ------------------------------------- #
# 价值网络
# ------------------------------------- #
 
class QValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions, n_layers):
        super(QValueNet, self).__init__()
        # 
        self.fc1 = nn.Linear(n_states + n_layers, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, n_hiddens)
        self.fc4 = nn.Linear(n_hiddens, 1)
        # self.fc1 = nn.Linear(n_states + n_layers, n_hiddens)
        # self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        # self.fc3 = nn.Linear(n_hiddens, 1)
    # 前向传播
    def forward(self, x, a):
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)  # [b, n_states + n_actions]
        x = self.fc1(cat)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # -->[b, n_layers]
        x = F.relu(x)
        x = self.fc4(x)  # -->[b, n_layers]
        # cat = torch.cat([x, a], dim=1)  # [b, n_states + n_actions]
        # x = self.fc1(cat)  # -->[b, n_hiddens]
        # x = F.relu(x)
        # x = self.fc2(x)  # -->[b, n_hiddens]
        # x = F.relu(x)
        # x = self.fc3(x)  # -->[b, 1]
        return x
 
# ------------------------------------- #
# 算法主体
# ------------------------------------- #
 
class DDPG:
    def __init__(self, n_states, n_hiddens, n_actions,n_layers,
                 sigma, actor_lr, critic_lr, tau, gamma, device):
 
        # 策略网络--训练
        self.actor = PolicyNet(n_states, n_hiddens, n_actions,n_layers).to(device)
        # 价值网络--训练
        self.critic = QValueNet(n_states, n_hiddens, n_actions,n_layers).to(device)
        # 策略网络--目标
        self.target_actor = PolicyNet(n_states, n_hiddens, n_actions,n_layers).to(device)
        # 价值网络--目标
        self.target_critic = QValueNet(n_states, n_hiddens, n_actions,n_layers).to(device)
        # 初始化价值网络的参数，两个价值网络的参数相同
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化策略网络的参数，两个策略网络的参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())
 
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
 
        # 属性分配
        self.gamma = gamma  # 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差，均值设为0
        self.tau = tau  # 目标网络的软更新参数
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.device = device
 
    # 动作选择
    def take_action(self, state):
        # 维度变换 list[n_states]-->tensor[1,n_states]-->gpu
        state = torch.tensor(state, dtype=torch.float).view(1,-1).to(self.device)
        # 策略网络计算出当前状态下的动作价值 [1,n_states]-->[1,1]-->int
        action = self.actor(state)
        action = action.detach().cpu().numpy().reshape((self.n_layers,))
        action = action + self.sigma * np.random.randn(self.n_layers)
        mean = action.mean()
        action = action - mean
        # print(action)
        return action
    
    # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data*self.tau)
 
    # 训练
    def update(self, transition_dict):
        # 从训练集中取出数据
        states = torch.tensor(transition_dict['states'], dtype=torch.float).view(-1,MODEL_SIZE).to(self.device)  # [b,n_states]
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,self.n_layers).to(self.device)  # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).view(-1,MODEL_SIZE).to(self.device)  # [b,next_states]
        
        # 价值目标网络获取下一时刻的动作[b,n_states]-->[b,n_actors]
        next_q_values = self.target_actor(next_states)
        # 策略目标网络获取下一时刻状态选出的动作价值 [b,n_states+n_actions]-->[b,1]
        next_q_values = self.target_critic(next_states, next_q_values)
        # 当前时刻的动作价值的目标值 [b,1]
        q_targets = rewards + self.gamma * next_q_values
        
        # 当前时刻动作价值的预测值 [b,n_states+n_actions]-->[b,1]
        q_values = self.critic(states, actions)
 
        # 预测值和目标值之间的均方差损失
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 价值网络梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
 
        # 当前状态的每个动作的价值 [b, n_actions]
        actor_q_values = self.actor(states)
        # 当前状态选出的动作价值 [b,1]
        score = self.critic(states, actor_q_values)
        # 计算损失
        actor_loss = -torch.mean(score)
        # 策略网络梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
 
        # 软更新策略网络的参数  
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络的参数
        self.soft_update(self.critic, self.target_critic)

 
class Env:
    def __init__(self,num_layers,):
        self.device = torch.device('cuda:3')
        # 经验回放池实例化
        self.replay_buffer = ReplayBuffer(capacity = 1000)
        self.episode_return = 0
        # 模型实例化
        self.agent = DDPG(n_states = MODEL_SIZE,  # 状态数
                    n_hiddens = 32,  # 隐含层数
                    n_actions = 2,  # 动作数
                    n_layers = num_layers, 
                    sigma = 0,  # 高斯噪声
                    actor_lr = 1e-3,  # 策略网络学习率
                    critic_lr = 1e-3,  # 价值网络学习率
                    tau = 1e-2,  # 软更新系数
                    gamma = 0.2,  # 折扣因子
                    device = self.device
                    )

        self.return_list = []  # 记录每个回合的return

    def get_action(self,state):
        # 获取当前状态对应的动作
        action = self.agent.take_action(state)
        return action
    
    def refresh_state(self,state,next_state,action,reward):
        # 更新经验回放池
        for _ in range(1):
            self.replay_buffer.add(state, action, reward, next_state, )
        # 累计每一步的reward
        self.episode_return += reward
        # self.episode_return = reward
 
        for _ in range(10):
            # 经验池随机采样batch_size组
            s, a, r, ns = self.replay_buffer.sample(min(self.replay_buffer.size(),1))
            # 构造数据集
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
            }
            # 模型训练
            self.agent.update(transition_dict)
    
        # 保存每一个回合的回报
        self.return_list.append(self.episode_return)

        # 打印回合信息
        print(f'iter:, return:{self.episode_return}, mean_return:{np.mean(self.return_list[-10:])}')
        return self.episode_return
 
 
 

