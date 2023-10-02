#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from copy import deepcopy
from model import make_model
import time
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from torch.autograd import Variable 
from config_reader import NODES,LEARNING_RATE,META_LEARNING_RATE,REP_LEARNING_RATE,NUM_FOR_PREDICT,LEN_INPUT,IN_CHANNELS,NB_BLOCK,NB_CHEV_FILTER,NB_TIME_FILTER,\
    BATCH_SIZE,K,NUM_OF_WEEKS,NUM_OF_DAYS,NUM_OF_HOURS,TIME_STRIDES,LOSS_FUNCTION,METRIC_METHOD,MISSING_VALUE,TEST_UPDATE_STEP,REPTILE_INNER_STEP,BATCH_INDEX,\
    MODEL_SIZE

class Graph_Client_FedAsync(nn.Module):
    def __init__(self,city,device_num,mode):
        super(Graph_Client_FedAsync, self).__init__()
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

        self.train_loader, self.train_target_tensor,self.support_loader, self.support_target_tensor,self.query_loader, self.query_target_tensor, self.val_loader, self.val_target_tensor, self.test_loader, self.test_target_tensor, self._mean, self._std,self.train_target_max, self.train_target_min = load_graphdata_channel1(self.graph_signal_matrix_filename, self.in_channels, self.num_of_hours, self.num_of_days, self.num_of_weeks, self.device, self.batch_size)
        self.adj_mx, self.distance_mx = get_adjacency_matrix(self.adj_filename, self.num_of_vertices, self.id_filename)
        self.net = make_model(self.device, self.nb_block, self.in_channels, self.K, self.nb_chev_filter, self.nb_time_filter, self.time_strides, self.distance_mx, self.num_for_predict, self.len_input, self.num_of_vertices)
        # self.net = make_model(self.device, 4, IN_CHANNELS, 1, 64, self.distance_mx, 8, NUM_OF_WEEKS,
        #        NUM_OF_DAYS, NUM_OF_HOURS, TIME_STRIDES, 12, dropout=.0, aware_temporal_context=True,
        #        ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True, use_LayerNorm=True)
        self.criterion = nn.L1Loss().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.learning_rate)
        self.fomaml_outer_optimizer = optim.Adam(self.net.parameters(), lr = self.meta_learning_rate)
        self.rep_outer_optimizer = optim.Adam(self.net.parameters(), lr = self.rep_learning_rate)
        self.time = 0
        self.epoch = 0

    def forward(self):
        pass

    def refresh(self,model):
        for w,w_t in zip(self.net.parameters(),model.parameters()):
            w.data.copy_(w_t.data)
            
    def Client_Train(self,):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 20
        model_size = MODEL_SIZE
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        for epoch in range(1):
            self.net.train()  # ensure dropout layers are in train mode
            for batch_index, batch_data in enumerate(self.train_loader):
                encoder_inputs, labels = batch_data
                self.optimizer.zero_grad()
                outputs = self.net(encoder_inputs,)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if batch_index > BATCH_INDEX:
                    break
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 1000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all

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
        # for _ in range(self.test_update_step):
        #     for batch_data in self.val_loader:
        #         encoder_inputs, labels = batch_data
        #         self.optimizer.zero_grad()
        #         outputs = self.net(encoder_inputs)
        #         loss = self.criterion(outputs, labels)
        #         loss.backward()
        #         self.optimizer.step()
                # d_lr.step()
        test_result = predict_and_save_results_mstgcn(self.net, self.test_loader, self.test_target_tensor, self.metric_method,self._mean, self._std, self.train_target_max, self.train_target_min, self.actual_node_number)
        test_result = np.array(test_result)
        return test_result
    
    def Client_Test_new(self,):
        '''
        :param global_step: int
        :param data_loader: torch.utils.data.utils.DataLoader
        :param data_target_tensor: tensor
        :param mean: (1, 1, 3, 1)
        :param std: (1, 1, 3, 1)
        :param type: string
        :return:
        '''
        for _ in range(self.test_update_step):
            for batch_data in self.val_loader:
                encoder_inputs, labels = batch_data
                self.optimizer.zero_grad()
                outputs = self.net(encoder_inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # d_lr.step()
        test_result = predict_and_save_results_mstgcn(self.net, self.test_loader, self.test_target_tensor, self.metric_method,self._mean, self._std, self.train_target_max, self.train_target_min, self.actual_node_number)

        return test_result

    def Client_Meta_Train(self,refresh_layers):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 2
        model_size = MODEL_SIZE
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])
        for index,param in enumerate(self.net.parameters()):
            if refresh_layers[index] < 0:
                param.requires_grad = False
            else:
                param.requires_grad = True

        start = time.time()
        for epoch in range(1):
            self.net.train()
            for batch_index, batch_data in enumerate(self.support_loader):
                encoder_inputs, labels = batch_data
                self.optimizer.zero_grad()
                outputs = self.net(encoder_inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if batch_index > BATCH_INDEX:
                    break

            for batch_index, batch_data in enumerate(self.query_loader):
                encoder_inputs, labels = batch_data
                self.fomaml_outer_optimizer.zero_grad()
                outputs = self.net(encoder_inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.fomaml_outer_optimizer.step()
                # d_meta_lr.step()
                break
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 1000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all

    def Client_Meta_Train_old(self):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 2
        model_size = MODEL_SIZE
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])
        
        start = time.time()
        for epoch in range(1):
            self.net.train()
            for batch_index, batch_data in enumerate(self.support_loader):
                encoder_inputs, labels = batch_data
                self.optimizer.zero_grad()
                outputs = self.net(encoder_inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if batch_index > BATCH_INDEX:
                    break

            for batch_index, batch_data in enumerate(self.query_loader):
                encoder_inputs, labels = batch_data
                self.fomaml_outer_optimizer.zero_grad()
                outputs = self.net(encoder_inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.fomaml_outer_optimizer.step()
                # d_meta_lr.step()
                break
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 1000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all

    def Client_Rep_Train(self,):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 20
        model_size = MODEL_SIZE
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        net_tem = deepcopy(self.net)
        optimizer = optim.Adam(net_tem.parameters(), lr = self.learning_rate)
        for epoch in range(REPTILE_INNER_STEP):
            self.net.train()
            for batch_index, batch_data in enumerate(self.train_loader):
                encoder_inputs, labels = batch_data
                optimizer.zero_grad()
                outputs = net_tem(encoder_inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if batch_index > BATCH_INDEX:
                    break
            
            self.rep_outer_optimizer.zero_grad()
            for w, w_t in zip(self.net.parameters(), net_tem.parameters()):
                if w.grad is None:
                    w.grad = Variable(torch.zeros_like(w)).to(self.device)
                w.grad.data.add_(w.data - w_t.data)

            self.rep_outer_optimizer.step()
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 1000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all
    
    def Client_Prox_Train(self,):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 20
        model_size = MODEL_SIZE
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        mu = 0.01
        global_model = deepcopy(self.net)
        # d_lr = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.1, last_epoch=-1, verbose=False)
        for epoch in range(1):
            self.net.train()
            for batch_index, batch_data in enumerate(self.train_loader):
                encoder_inputs, labels = batch_data
                self.optimizer.zero_grad()
                outputs = self.net(encoder_inputs)
                proximal_term = 0.01
                for w, w_t in zip(self.net.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = self.criterion(outputs, labels) + (mu / 2) *  proximal_term
                loss.backward()
                self.optimizer.step()
                # d_lr.step()
                if batch_index > BATCH_INDEX:
                    break
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 1000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all

    def Client_pFedMe_Train(self):
        # 基础上传和下载速率（KB/s）
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 20
        model_size = MODEL_SIZE
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        model_tem = deepcopy(self.net)
        optim_tem = torch.optim.Adam(model_tem.parameters(), lr = self.learning_rate)
        for _ in range(1):
            for batch_index, batch_data in enumerate(self.train_loader):
                encoder_inputs, labels = batch_data
                optim_tem.zero_grad()
                output = model_tem(encoder_inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                optim_tem.step()
                if batch_index > BATCH_INDEX:
                    break
        
        for tem_0, tem in zip(model_tem.parameters(), self.net.parameters()):
            tem.data = tem.data - 1.0 * self.rep_learning_rate * (tem.data - tem_0.data)
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 1000
        t_c = np.array(end-start) * 100
        t_all = t_c + tmp_tranmission
        self.time = t_all

    def Local_FedAsyED_Train(self,):
        self.old_model = deepcopy(self.net)
        self.tem_model = deepcopy(self.net)
        base_link_rate = np.random.randint(1300, 4500, (2,))
        # 通信延迟
        link_delay = np.random.randint(1, 10, (2,))
        # 实际上传下载速率
        actual_link_rate = base_link_rate + link_delay * 200
        model_size = MODEL_SIZE
        # 通信时间：参数数量 * 4(32位浮点数占4字节) / 1024(千字节) / 通信速率
        tranmission_time = model_size * 4 / 1024 / actual_link_rate
        tmp_tranmission = sum(tranmission_time[0:2,])

        start = time.time()
        for _ in range(self.update_step):
            for index,support in enumerate(self.support_loader):
                support_x, support_y = support
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                self.optim.zero_grad()
                output = self.net(support_x)
                output = torch.flatten(output)
                loss = self.loss_function(output, support_y)
                loss.backward()
                self.optim.step()
                # if index > BATCH_INDEX:
                #     break

        for w,w_t,w_s in zip(self.net.parameters(),self.old_model.parameters(), self.tem_model.parameters()):
            if w.grad is None:
                w.grad = Variable(torch.zeros_like(w)).to(self.device)
            w_s.data.add_(w.data - w_t.data)
        end = time.time()
        tmp_tranmission = np.array(tmp_tranmission) * 1000
        t_c = np.array(end - start) * 100
        t_all = t_c + tmp_tranmission

        self.time = t_all
