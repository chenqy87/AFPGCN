import os
import torch
import numpy as np
from model import make_model
import torch.nn as nn
import pandas as pd
import os
from Graph_Client import Graph_Client
from lib.utils_tem import get_adjacency_matrix
from torch.utils.tensorboard import SummaryWriter
import time
from config_reader import IN_CHANNELS,NB_BLOCK,K,NB_CHEV_FILTER,NB_TIME_FILTER,TIME_STRIDES,NUM_FOR_PREDICT,LEN_INPUT,NODES,TEST_UPDATE_STEP,BATCH_SIZE,BATCH_INDEX

class Local_Graph(nn.Module):
  def __init__(self, device_num,number):
    super(Local_Graph, self).__init__()
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
    self.net = make_model(self.device, self.nb_block, self.in_channels, self.K, self.nb_chev_filter, self.nb_time_filter, self.time_strides, self.adj_mx, self.num_for_predict, self.len_input, self.num_of_vertices)

    self.writer = SummaryWriter('./results/s_{}_b_{}_c_{}_k_{}_i_{}_n_{}/Local_Graph'.format(TEST_UPDATE_STEP,BATCH_SIZE,IN_CHANNELS,self.K,BATCH_INDEX,self.number))
    self.clients = []
    self.folder = r"./data/applied_data"
    folder_city_set = os.listdir(self.folder)
    folder_city_set = [i.split("_")[0] for i in folder_city_set]
    for city in folder_city_set:
      self.clients.append(Graph_Client(city,device_num,"train"))

  def forward(self):
    pass

  def Local_Train(self):
    id_train_0 = list(range(len(self.clients)))
    for id,j in enumerate(id_train_0):
      self.clients[j].Client_Train()

  def Local_Test(self,round,num):
    # id_test = list(range(len(self.test_clients)))
    id_test = list(range(len(self.clients)))
    mae_list = []
    rmse_list = []
    mape_list = []
    r2_list = []
    for a,id in enumerate(id_test):
        # self.test_clients[id].refresh(self.net)
        # test_result = self.test_clients[id].Client_Test()
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
