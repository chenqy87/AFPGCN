from FedBuff_Graph import FedBuff_Graph
import pandas as pd
from Finish_Email import send_an_finish_message
import torch
import os
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


from config_reader import TEST_UPDATE_STEP,RESULT_SAVE_NUMBER_START,RESULT_SAVE_NUMBER_END,TEST_FREQUENCY,TRAINING_EPOCHS,BATCH_SIZE,IN_CHANNELS,K,BATCH_INDEX

def main(model_name,device_num,epoch,num):
    folder = r"./model/model_s_{}_b_{}_c_{}_k_{}_i_{}_{}".format(TEST_UPDATE_STEP,BATCH_SIZE,IN_CHANNELS,K,BATCH_INDEX,num)
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = os.path.join(folder,model_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    fed_net = FedBuff_Graph(device_num,num)
    for i in range(1,1+epoch):
        print("{} round training.".format(i))
        if i == 1:
            fed_net.Fed_Test(0,num) 
            torch.save({'model': fed_net.state_dict()},os.path.join(folder,"model_epoch_{}.pth".format(0)))
        fed_net.Fed_Train(epoch)
        if i%TEST_FREQUENCY == 0:
            fed_net.Fed_Test(i,num) 
            torch.save({'model': fed_net.state_dict()},os.path.join(folder,"model_epoch_{}.pth".format(i)))

if __name__ == '__main__':
    model_name = 'FedBuff_Graph'
    device_num = 2
    epoch = TRAINING_EPOCHS
    for i in range(RESULT_SAVE_NUMBER_START,RESULT_SAVE_NUMBER_END):
        main(model_name,device_num,epoch,i)
    send_an_finish_message(os.path.basename(__file__))