from Graph_Client_AsyFedMeta import Graph_Client_AsyFedMeta
import os
import torch
import numpy as np
from model import make_model
from torch.autograd import Variable
import torch.nn as nn
import os
from lib.utils import get_adjacency_matrix
import time
from torch.utils.tensorboard import SummaryWriter
from DDPG import Env
from copy import deepcopy
from config_reader import IN_CHANNELS,NB_BLOCK,K,NB_CHEV_FILTER,NB_TIME_FILTER,TIME_STRIDES,NUM_FOR_PREDICT,LEN_INPUT,NODES,TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX

class AsyFedMeta_Graph(nn.Module):
  def __init__(self, device_num,number):
    super(AsyFedMeta_Graph, self).__init__()
    device = torch.device('cuda:{}'.format(device_num))
    self.device = device
    self.number = number
    self.in_channels = IN_CHANNELS
    self.nb_block = NB_BLOCK
    self.K = K
    self.nb_chev_filter = NB_CHEV_FILTER
    self.nb_time_filter = NB_TIME_FILTER
    self.time_strides = TIME_STRIDES
    self.num_for_predict = NUM_FOR_PREDICT
    self.len_input = LEN_INPUT
    self.num_of_vertices = NODES

    self.adj_mx, self.distance_mx = get_adjacency_matrix(r'./data/district_relationship/None.csv', self.num_of_vertices, None)
    self.deep_frequency = 5
    self.net = make_model(self.device, self.nb_block, self.in_channels, self.K, self.nb_chev_filter, self.nb_time_filter, self.time_strides, self.distance_mx, self.num_for_predict, self.len_input, self.num_of_vertices)
    self.old_net = deepcopy(self.net)
    self.writer = SummaryWriter('./results/s_{}_b_{}_c_{}_k_{}_i_{}_n_{}/AsyFedMeta_Graph'.format(TEST_UPDATE_STEP,BATCH_SIZE,IN_CHANNELS,self.K,BATCH_INDEX,self.number))
    self.clients = []
    self.folder = r"./data/applied_data"
    folder_city_set = os.listdir(self.folder)
    folder_city_set = [i.split("_")[0] for i in folder_city_set]
    for id_0,city in enumerate(folder_city_set):
      self.clients.append(Graph_Client_AsyFedMeta(id_0,city,device_num,"train"))

    layer = 0
    for index,param in enumerate(self.net.parameters()):
      layer += 1
    self.num_layers = layer
    self.env = Env(self.num_layers)
    self.refresh_layers = np.random.randn(self.num_layers,)
    self.reward = 0
    self.old_r2 = 0
    self.index = 0
    self.arg_index = self.refresh_layers >= 0
    # self.old_mape = 0

  def forward(self):
    pass

  def Fed_Train(self,round):
    self.old_net = deepcopy(self.net)
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      if self.clients[j].time <= 0:
        self.clients[j].local_initialization(self.net)
        self.clients[j].Client_Meta_Train(self.refresh_layers)
        self.clients[j].epoch = round
      else:
        continue

    id_train_0 = list(range(len(self.clients)))
    time_start = time.time()
    id_train = []
    for id in id_train_0:
      self.clients[id].time = max(self.clients[id].time - 100,0)
      if self.clients[id].time <= 0:
        id_train.append(id)

    weight = []
    for id,j in enumerate(id_train):
      # weight.append(np.power(np.exp(1),self.clients[j].epoch-round))
      weight.append(1)
    weight = np.array(weight)
    weight = weight / weight.sum()
    
    with torch.no_grad():
      b = 0
      for id,j in enumerate(id_train):
        a = 0
        for w,w_t in zip(self.net.parameters(),self.clients[j].net.parameters()):
          layer_refresh_status = self.refresh_layers[a]
          a += 1
          if layer_refresh_status < 0:
              continue
          if (w is None or id == 0):
            w_tem = Variable(torch.zeros_like(w)).to(self.device)
            w.data.copy_(w_tem.data)
          if w_t is None:
            w_t = Variable(torch.zeros_like(w)).to(self.device)
          w.data.add_(w_t.data*weight[id])

    time_end = time.time()
    time_total = time_end - time_start + 30
    self.writer.add_scalar('Time', time_total, round)
    self.writer.add_scalar('Num', len(id_train), round)

  def Fed_Test(self,round,num):
    self.input_rl_net = deepcopy(self.net)
    # id_test = list(range(len(self.test_clients)))
    id_test = list(range(len(self.clients)))
    mae_list = []
    rmse_list = []
    mape_list = []
    r2_list = []
    for a,id in enumerate(id_test):
        self.clients[id].refresh(self.net)
        test_result_set = self.clients[id].Client_Test()
        for j in range(self.clients[id].actual_node_number):
          test_result = test_result_set[j]
          test_mae, test_rmse, test_mape, test_r2 = test_result[0], test_result[1], test_result[2], test_result[3]
          self.writer.add_scalar('MAE_{}_{}'.format(id,j), test_mae, round)
          self.writer.add_scalar('RMSE_{}_{}'.format(id,j), test_rmse, round)
          self.writer.add_scalar('MAPE_{}_{}'.format(id,j), test_mape, round)
          self.writer.add_scalar('R2_{}_{}'.format(id,j), test_r2, round)
          mae_list.append(test_mae)
          rmse_list.append(test_rmse)
          mape_list.append(test_mape)
          r2_list.append(test_r2)
    mae_list = np.array(mae_list)
    rmse_list = np.array(rmse_list)
    mape_list = np.array(mape_list)
    r2_list = np.array(r2_list)
    mae_mean = mae_list.mean()
    rmse_mean = rmse_list.mean()
    mape_mean = mape_list.mean()
    r2_mean = r2_list.mean()
    print("Round {}:\n".format(round))
    print('all MAE: %.2f' % (mae_mean))
    print('all RMSE: %.2f' % (rmse_mean))
    print('all MAPE: %.2f' % (mape_mean))
    print('all R2: %.2f' % (r2_mean))

    self.writer.add_scalar('MAE', mae_mean, round)
    self.writer.add_scalar('RMSE', rmse_mean, round)
    self.writer.add_scalar('MAPE', mape_mean, round)
    self.writer.add_scalar('R2', r2_mean, round)

    if round == 0:
      self.old_input_rl_net = deepcopy(self.input_rl_net)
      for w,w_t in zip(self.old_input_rl_net.parameters(),self.input_rl_net.parameters()):
        w_t = Variable(torch.zeros_like(w)).to(self.device)
        w = Variable(torch.zeros_like(w)).to(self.device)
    else:
      self.old_input_rl_net = deepcopy(self.input_rl_net)
      self.input_rl_net = deepcopy(self.net)
    for w,w_t,w_rl in zip(self.net.parameters(),self.old_net.parameters(),self.input_rl_net.parameters()):
      w_rl.data = w.data - w_t.data

    self.reward += (r2_mean - self.old_r2) * 10
    
    if round > 0:
      self.reward += r2_mean
      print('Reward: %.2f' % (self.reward))
      
      total_reward = self.env.refresh_state(self.model_1d(self.old_input_rl_net),self.model_1d(self.input_rl_net),self.refresh_layers,self.reward)
      if round <= 100:
        self.refresh_layers = np.ones((self.num_layers,))
      else:
        self.refresh_layers = self.env.agent.take_action(self.model_1d(self.net))
        self.old_r2 = r2_mean

      # self.old_mape = mape_mean
      # print(self.refresh_layers)
      self.writer.add_scalar('RL_total_reward', total_reward, round)
      self.writer.add_scalar('RL_reward', self.reward, round)

  def model_1d(self,model):
    for index,param in enumerate(model.parameters()):
      param = torch.flatten(param)
      if index == 0:
        params = Variable(torch.zeros_like(param)).to(self.device)
        params.copy_(param.data)
      else:
        params = torch.cat([params,param.data], dim = 0)
    params = params
    return params