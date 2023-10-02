#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from copy import deepcopy
from sklearn.svm import SVR
from model import make_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from lib.metrics import masked_mape_np
from sklearn.metrics import r2_score
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from torch.autograd import Variable 
from config_reader import NODES,LEARNING_RATE,META_LEARNING_RATE,REP_LEARNING_RATE,NUM_FOR_PREDICT,LEN_INPUT,IN_CHANNELS,NB_BLOCK,NB_CHEV_FILTER,NB_TIME_FILTER,\
    BATCH_SIZE,K,NUM_OF_WEEKS,NUM_OF_DAYS,NUM_OF_HOURS,TIME_STRIDES,LOSS_FUNCTION,METRIC_METHOD,MISSING_VALUE,TEST_UPDATE_STEP,REPTILE_INNER_STEP,BATCH_INDEX

class SVR_Client(nn.Module):
    def __init__(self,city,device_num,mode):
        super(SVR_Client, self).__init__()
        self.adj_filename = r"./data/district_relationship/{}.csv".format(city)
        if mode == 'train':
            self.graph_signal_matrix_filename = r"./data/applied_data/{}.npz".format(city)
        elif mode == "test":
            self.graph_signal_matrix_filename = r"./data/applied_data/{}.npz".format(city)
    
        actual_data = pd.read_csv(r"./data/city_clients.csv")
        
        self.actual_node_number = actual_data[actual_data["City"] == city]["Clients"].tolist()[0]
        self.num_of_vertices = NODES
        self.num_for_predict = NUM_FOR_PREDICT
        self.len_input = LEN_INPUT
        self.dataset_name = city
        self.id_filename = None

        self.in_channels = IN_CHANNELS
        self.nb_block = NB_BLOCK
        self.K = K
        self.nb_chev_filter = NB_CHEV_FILTER
        self.nb_time_filter = NB_TIME_FILTER
        self.batch_size = BATCH_SIZE
        self.num_of_weeks = NUM_OF_WEEKS
        self.num_of_days = NUM_OF_DAYS
        self.num_of_hours = NUM_OF_HOURS
        self.learning_rate = LEARNING_RATE
        self.metric_method = METRIC_METHOD
        self.missing_value = MISSING_VALUE
        self.time_strides = TIME_STRIDES
        self.meta_learning_rate = META_LEARNING_RATE
        self.rep_learning_rate = REP_LEARNING_RATE
        self.test_update_step = TEST_UPDATE_STEP

        device = torch.device('cuda:{}'.format(device_num))
        self.device = device

        self.train_loader, self.train_target_tensor,self.support_loader, self.support_target_tensor,self.query_loader, self.query_target_tensor, self.val_loader, self.val_target_tensor, self.test_loader, self.test_target_tensor, self._mean, self._std,self.train_target_mean, self.train_target_std = load_graphdata_channel1(self.graph_signal_matrix_filename, self.in_channels, self.num_of_hours, self.num_of_days, self.num_of_weeks, self.device, self.batch_size)
        self.adj_mx, self.distance_mx = get_adjacency_matrix(self.adj_filename, self.num_of_vertices, self.id_filename)
        self.net = SVR(kernel='sigmoid')


        self.criterion = nn.L1Loss().to(self.device)

    def forward(self):
        pass

    def Client_Train(self,):
        for epoch in range(1):
            for batch_index, batch_data in enumerate(self.train_loader):
                encoder_inputs, labels = batch_data
                for i in range(labels.shape[1]):
                    input_0 = encoder_inputs[:,i,0,:].cpu().numpy()
                    output_0 = labels[:,i,0].cpu().numpy()
                    if i == 0:
                        input_all = deepcopy(input_0)
                        output_all = deepcopy(output_0)
                    else:
                        input_all = np.concatenate((input_all,input_0),axis = 0)
                        output_all = np.concatenate((output_all,output_0),axis = 0)
                self.net.fit(input_all,output_all)
                break

    def Client_Test(self,):
        '''
        :param global_step: int
        :param data_loader: torch.utils.data.utils.DataLoader
        :param data_target_tensor: tensor
        :param mean: (1, 1, 3, 1)
        :param std: (1, 1, 3, 1)
        :param type: string
        :return:
        '''
        excel_list = []
        for batch_index, batch_data in enumerate(self.test_loader):
            encoder_inputs, labels = batch_data
            for i in range(labels.shape[1]):
                input_0 = encoder_inputs[:,i,0,:].cpu().numpy()
                output_0 = labels[:,i,0].cpu().numpy()
                if i == 0:
                    input_all = deepcopy(input_0)
                    output_all = deepcopy(output_0)
                else:
                    input_all = np.concatenate((input_all,input_0),axis = 0)
                    output_all = np.concatenate((output_all,output_0),axis = 0)
        results = self.net.predict(input_all)
        # results = results * self._mean[0,0,0,0] + self._std[0,0,0,0]
        # output_all = output_all  * self._mean[0,0,0,0] + self._std[0,0,0,0]
        mae = mean_absolute_error(output_all, results)
        rmse = mean_squared_error(output_all, results) ** 0.5
        mape = masked_mape_np(output_all, results, 0)
        r2 = r2_score(output_all, results)
        excel_list.extend([mae, rmse, mape, r2])
        test_result = np.array(excel_list)
        return test_result
    