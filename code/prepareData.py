import os
import numpy as np
import argparse
import configparser
from config_reader import TRAINING_PERCENT,VALIDATION_PERCENT,NODES,NUM_FOR_PREDICT,POINTS_PER_HOUR,NUM_OF_HOURS,IN_CHANNELS
import matplotlib.pyplot as plt


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = label_start_idx
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)
    num_of_weeks, num_of_days, num_of_hours: int
    label_start_idx: int, the first index of predicting target, 预测值开始的那个点
    num_for_predict: int,
                     the number of points will be predicted for each sample
    points_per_hour: int, default 12, number of points per hour
    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)
    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)
    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)
    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''
    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(graph_signal_matrix_filename,
                                                     num_of_weeks, num_of_days,
                                                     num_of_hours, num_for_predict,
                                                     points_per_hour=12, save=False):
    '''
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    graph_signal_matrix_filename = graph_signal_matrix_filename.replace('kwh_data',"kwh_data_1")
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (sequence_length, num_of_vertices, num_of_features)

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(
            sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]


    split_line1 = int(len(all_samples) * TRAINING_PERCENT)
    split_line2 = int(len(all_samples) * VALIDATION_PERCENT)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)
    train_x_norm = np.nan_to_num(train_x_norm)
    val_x_norm = np.nan_to_num(val_x_norm)
    test_x_norm = np.nan_to_num(test_x_norm)
    # print(graph_signal_matrix_filename)
    # print(real_nodes)
    # mean = stats["_mean"].mean(axis=(0,1,3), keepdims=True)
    # for i in range(real_nodes,15):
    #     train_x_norm[:,i:i+1,:,:] = train_x_norm[:,i-real_nodes:i-real_nodes+1,:,:]
    #     val_x_norm[:,i:i+1,:,:] = val_x_norm[:,i-real_nodes:i-real_nodes+1,:,:]
    #     test_x_norm[:,i:i+1,:,:] = test_x_norm[:,i-real_nodes:i-real_nodes+1,:,:]
    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape,)
    print('train data _std :', stats['_std'].shape,)

    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        save_province_path = './data/kwh_data_2/{}'.format(province)
        # save_graph_signal_matrix_filename = './data/kwh_data_1/{}/{}.npz'.format(province,city)
        if not os.path.exists(save_province_path):
            os.mkdir(save_province_path)
        filename = os.path.join(save_province_path, file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_astcgn'
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                            )
    return all_data


def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    # train_feature_0 = train[:,:,:1,:]
    train_feature_1 = train[:,:,1:,:]
    val_feature_1 = val[:,:,1:,:]
    test_feature_1 = test[:,:,1:,:]
    mean_feature_1 = train_feature_1.mean(axis=(0,1,2,3), keepdims=True)
    std_feature_1 = train_feature_1.std(axis=(0,1,2,3), keepdims=True)
    mean = train.mean(axis=(0,3), keepdims=True)
    std = train.std(axis=(0,3), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std
    
    def normalize_1(x):
        return (x - mean_feature_1) / std_feature_1

    train_norm = normalize(train)[:,:,:1,:]
    val_norm = normalize(val)[:,:,:1,:]
    test_norm = normalize(test)[:,:,:1,:]

    train_feature_norm = normalize_1(train_feature_1)
    val_feature_norm = normalize_1(val_feature_1)
    test_feature_norm = normalize_1(test_feature_1)

    train_norm = np.dstack((train_norm,train_feature_norm))
    val_norm = np.dstack((val_norm,val_feature_norm))
    test_norm = np.dstack((test_norm,test_feature_norm))

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm

def generate_and_concat_data(province,city):
    graph_signal_matrix_filename = './data/kwh_data/{}/{}.npz'.format(province,city)
    save_province_path = './data/kwh_data_1/{}'.format(province)
    save_graph_signal_matrix_filename = './data/kwh_data_1/{}/{}.npz'.format(province,city)
    if not os.path.exists(save_province_path):
        os.mkdir(save_province_path)
    # data_seq = np.load(graph_signal_matrix_filename)['data'][:,:,:IN_CHANNELS]  # (sequence_length, num_of_vertices, num_of_features)
    data_seq = np.load(graph_signal_matrix_filename)['data'] # (sequence_length, num_of_vertices, num_of_features)
    data_seq_1 = data_seq[:,:,:1]
    data_seq_2 = data_seq[:,:,1:]  #修改输入通道
    data_seq = np.dstack((data_seq_1,data_seq_2,))
    data_seq = data_seq[:,:,:IN_CHANNELS]
    print(data_seq)
    # data_seq = data_seq / 1000
    sequence_length = data_seq.shape[0]
    num_of_vertices = data_seq.shape[1]
    num_of_features = data_seq.shape[2]
    blank_vertices = NODES - num_of_vertices
    print()
    blank_data = np.zeros(shape = (sequence_length,blank_vertices,num_of_features))
    data = np.concatenate([data_seq,blank_data],axis = 1)
    for i in range(num_of_vertices,NODES):
        data[:,i:i+1,:] = data[:,i-num_of_vertices:i-num_of_vertices+1,:]
    # for i in range(num_of_vertices):
    #     tem_data = data[:,i:i+1,:1].reshape(-1)
    #     plt.plot(tem_data)
    #     plt.show()
    data = np.nan_to_num(data)
    np.savez(save_graph_signal_matrix_filename,data = data)
    # return num_of_vertices

def generate_data_city(province,city):
    graph_signal_matrix_filename = './data/kwh_data/{}/{}.npz'.format(province,city)
    points_per_hour = POINTS_PER_HOUR
    num_for_predict = NUM_FOR_PREDICT
    num_of_hours = NUM_OF_HOURS
    print(graph_signal_matrix_filename)
    generate_and_concat_data(province,city)
    
    all_data = read_and_generate_dataset(graph_signal_matrix_filename, 0, 0, num_of_hours, num_for_predict, points_per_hour=points_per_hour, save=True)

folder = './data/district_relationship_1'
province_set = os.listdir(folder)
province_set.remove("None.csv")
for province in province_set:
    province_folder = os.path.join(folder,province)
    city_set = os.listdir(province_folder)
    city_set = [i.split(".")[0] for i in city_set]
    for city in city_set:
        generate_data_city(province,city)
