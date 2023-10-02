from Local_Graph import Local_Graph
import pandas as pd
from Finish_Email import send_an_finish_message
import os
from config_reader import TEST_UPDATE_STEP,RESULT_SAVE_NUMBER_START,RESULT_SAVE_NUMBER_END,TEST_FREQUENCY,TRAINING_EPOCHS,BATCH_SIZE,IN_CHANNELS,K,BATCH_INDEX

def main(model_name,device_num,epoch,num):
    folder = r"./model/model_s_{}_b_{}_c_{}_k_{}_i_{}_{}".format(TEST_UPDATE_STEP,BATCH_SIZE,IN_CHANNELS,K,BATCH_INDEX,num)
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = os.path.join(folder,model_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    fed_net = Local_Graph(device_num,num)
    for i in range(1,1+epoch):
        print("{} round training.".format(i))
        if i == 1:
            fed_net.Local_Test(0,num) 
        fed_net.Local_Train()
        if i%TEST_FREQUENCY == 0:
            fed_net.Local_Test(i,num) 

if __name__ == '__main__':
    model_name = 'Local_Graph'
    device_num = 1
    epoch = TRAINING_EPOCHS
    for i in range(RESULT_SAVE_NUMBER_START,RESULT_SAVE_NUMBER_END):
        main(model_name,device_num,epoch,i)
    send_an_finish_message(os.path.basename(__file__))