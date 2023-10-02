from SVR import SVR
import pandas as pd
from Finish_Email import send_an_finish_message
import torch
import os
from config_reader import TEST_UPDATE_STEP,RESULT_SAVE_NUMBER_START,RESULT_SAVE_NUMBER_END,TEST_FREQUENCY,TRAINING_EPOCHS,BATCH_SIZE,IN_CHANNELS,K,BATCH_INDEX

def main(model_name,device_num,epoch,num):
    fed_net = SVR(device_num,num)
    fed_net.Fed_Train() 
    fed_net.Fed_Test(0,num) 

if __name__ == '__main__':
    model_name = 'SVR'
    device_num = 2
    epoch = TRAINING_EPOCHS
    for i in range(RESULT_SAVE_NUMBER_START,RESULT_SAVE_NUMBER_END):
        main(model_name,device_num,epoch,i)
    send_an_finish_message(os.path.basename(__file__))