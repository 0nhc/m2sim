import argparse
import itertools
import logging
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm as tqdm_

from . import utils, structs, globals
from .modeling.vectornet import VectorNet
from .waymo_tutorial import _parse
from .dataset_waymo import get_instance

from sklearn.neighbors import NearestNeighbors

import yaml

import pickle
import copy
from IPython import embed

def init_config(args):
    '''from utils.py'''
    if args.config is not None:
        fs = open(os.path.join('.', 'configs', args.config))
        config = yaml.load(fs, yaml.FullLoader)
        for key in config:
            setattr(args, key, config[key])

    dic = {}
    for i, param in enumerate(args.other_params + args.eval_params + args.train_params):
        if '=' in param:
            index = str(param).index('=')
            key = param[:index]
            value = param[index + 1:]
            # key, value = param.split('=')
            dic[key] = value if not str(value).isdigit() else int(value)
        else:
            dic[param] = True
    args.other_params = dic
    # print(dict(sorted(vars(args_).items())))

    os.makedirs(args.log_dir, exist_ok=True)

def init_m2i_args():
    args = argparse.Namespace()
    args.do_train = False
    args.do_eval = True
    args.do_test = False

    ################
    # args.data_dir = ''
    # args.data_txt = ''
    # args.output_dir = ''
    # args.temp_file_dir = ''
    # args.train_batch_size = 1
    # args.eval_batch_size = 1
    # args.model_recover_path = ''
    # args.validation_model = 0
    # args.learning_rate = 0
    # args.weight_decay = 0
    # args.num_train_epochs = 0
    # args.seed = 0
    ################
    args.log_dir = 'log'
    args.no_cuda = False
    args.hidden_size = 128
    args.hidden_dropout_prob = 0.1
    args.sub_graph_depth = 3
    args.global_graph_depth = 1
    args.debug = False
    args.initializer_range = 0.02
    args.sub_graph_batch_size = 4096
    args.distributed_training = 0
    args.cuda_visible_device_num = None
    args.use_map = False
    args.reuse_temp_file = False
    args.old_version = False
    args.max_distance = 50.0
    args.no_sub_graph = False
    args.no_agents = False
    args.other_params = []
    args.eval_params = []
    args.train_params = []
    args.not_use_api = False
    args.core_num = 16
    args.visualize = False
    args.train_extra = False
    args.use_centerline = False
    args.autoregression = None
    args.lstm = False
    args.add_prefix = None
    args.attention_decay = False
    args.placeholder = 0.0
    args.multi = None
    args.method_span = [0, 1]
    args.nms_threshold = 7.2
    args.stage_one_K = None
    args.master_port = '12355'
    args.gpu_split = 0
    args.waymo = True
    args.argoverse = False
    args.nuscenes = False
    args.future_frame_num = 80
    args.future_test_frame_num = 16
    args.single_agent = True
    # args.agent_type = 'vehicle'
    args.inter_agent_types = None
    args.mode_num = 6
    args.joint_target_each = 80
    args.joint_target_type = 'no'
    args.joint_nms_type = 'and'
    args.debug_mode = True
    args.traj_loss_coeff = 1.0
    args.short_term_loss_coeff = 0.0
    args.classify_sub_goals = False
    args.config = 'conditional_pred_for_script.yaml'
    args.relation_file_path = ''
    args.influencer_pred_file_path = None
    args.relation_pred_file_path = None
    args.reverse_pred_relation = False
    args.eval_rst_saving_number = None
    args.eval_exp_path = 'eval_exp_path'
    args.infMLP = 8
    args.relation_pred_threshold = 0.8
    args.direct_relation_path = None
    args.all_agent_ids_path = None
    args.vehicle_r_pred_threshold = None
    
    init_config(args)

    utils.args = args

    return args

def init_m2i(ckpt_path, reactor_type='vehicle'):
    assert reactor_type in ['vehicle', 'pedestrian', 'cyclist']
    args = init_m2i_args()

    if reactor_type == 'vehicle':
        args.agent_type = 'vehicle'
        args.other_params['pair_vv'] = True
    elif reactor_type == 'pedestrian':
        args.agent_type = 'pedestrian'
        args.other_params['pair_vp'] = True
    elif reactor_type == 'cyclist':
        args.agent_type = 'cyclist'
        args.other_params['pair_vc'] = True
    else:
        assert NotImplementedError

    model = VectorNet(args).cuda()
    model.eval()
    model_recover = torch.load(ckpt_path)
    model.load_state_dict(model_recover)

    return args, model

def choose_reactor(input_data, args):
    '''生成从车'''
    # influencer
    assert (input_data['state/is_sdc'] == 1).sum() == 1
    influencer_idx = (input_data['state/is_sdc'] == 1).nonzero()[0][0]

    # reactor
    valid_threshold_past = 1
    valid_threshold_current = 1
    temp_valid_past = input_data['state/past/valid'].copy().sum(axis=-1) >= valid_threshold_past
    temp_valid_current = input_data['state/current/valid'].copy().sum(axis=-1) >= valid_threshold_current
    chosen_cars = np.logical_and(temp_valid_past, temp_valid_current)   # use logical_and to ensure current is valid. get_instance() requires this.
    chosen_cars_idx = chosen_cars.nonzero()[0]

    # relation
    if args.agent_type == 'vehicle':
        agent_pair_label = 1
    elif args.agent_type == 'pedestrian':
        agent_pair_label = 2
    elif args.agent_type == 'cyclist':
        agent_pair_label = 3
    else:
        raise NotImplementedError

    split_data_list = {}
    for reactor_idx in chosen_cars_idx:
        if influencer_idx == reactor_idx:
            continue

        # label the cars
        new_decoded_example = {}
        new_decoded_example.update(copy.deepcopy(input_data))
        new_decoded_example['state/objects_of_interest'] *= 0
        new_decoded_example['state/objects_of_interest'][influencer_idx] = 1
        new_decoded_example['state/objects_of_interest'][reactor_idx] = 1
        new_decoded_example['state/tracks_to_predict'] *= 0
        new_decoded_example['state/tracks_to_predict'][influencer_idx] = 1
        new_decoded_example['state/tracks_to_predict'][reactor_idx] = 1

        # move influencer and reactor to the first 2 place
        for k,v in new_decoded_example.items():
            if k.split('/')[0] == 'state' or k.split('/')[0] == 'dummy':
                if len(v.shape) == 2:
                    new_decoded_example[k][[influencer_idx,0],:] = v[[0,influencer_idx],:]
                    if reactor_idx == 0:
                        temp_reactor_idx = influencer_idx
                    else:
                        temp_reactor_idx = reactor_idx
                    new_decoded_example[k][[temp_reactor_idx,1],:] = v[[1,temp_reactor_idx],:]
                else:
                    new_decoded_example[k][[influencer_idx,0]] = v[[0,influencer_idx]]
                    if reactor_idx == 0:
                        temp_reactor_idx = influencer_idx
                    else:
                        temp_reactor_idx = reactor_idx
                    new_decoded_example[k][[temp_reactor_idx,1]] = v[[1,temp_reactor_idx]]

        # relation
        # interaction_label = 1 if influencer_idx > reactor_idx else 0
        # new_decoded_example['relation'] = np.array([influencer_idx, reactor_idx, interaction_label, agent_pair_label])
        inf_state_id = new_decoded_example['state/id'][0]
        rea_state_id = new_decoded_example['state/id'][1]
        interaction_label = 1 if inf_state_id > rea_state_id else 0
        new_decoded_example['relation'] = np.array([inf_state_id, rea_state_id, interaction_label, agent_pair_label])

        # filter out empty data to save memory space
        pastcurrent_valid = np.concatenate([new_decoded_example['state/past/valid'],new_decoded_example['state/current/valid']], axis=1)
        pastcurrent_valid_agent = pastcurrent_valid.sum(1)>0

        if not pastcurrent_valid_agent[0] == True and pastcurrent_valid_agent[1] == True:
            # embed(header='not pastcurrent_valid_agent[0] == True and pastcurrent_valid_agent[1] == True')
            raise NotImplementedError()
        assert pastcurrent_valid_agent[0] == True and pastcurrent_valid_agent[1] == True

        for k,v in new_decoded_example.items():
            if k.split('/')[0] == 'state' or k.split('/')[0] == 'dummy':
                new_decoded_example[k] = new_decoded_example[k][pastcurrent_valid_agent]

        new_decoded_example['influencer_reactor_idx'] = (influencer_idx, reactor_idx)
        
        if 'dummy/future/gt/x' not in new_decoded_example.keys():
            new_decoded_example['dummy/future/gt/x'] = copy.deepcopy(new_decoded_example['state/future/x'])
            new_decoded_example['dummy/future/gt/y'] = copy.deepcopy(new_decoded_example['state/future/y'])
        else:
            new_decoded_example['state/future/x'] = copy.deepcopy(new_decoded_example['dummy/future/gt/x'])#TODO
            new_decoded_example['state/future/y'] = copy.deepcopy(new_decoded_example['dummy/future/gt/y'])
            new_decoded_example['state/future/valid'] = (new_decoded_example['state/future/y']!=-1).astype(np.int)

        split_data_list[reactor_idx] = copy.deepcopy(new_decoded_example)

    return split_data_list, influencer_idx

def prepare_data(args, input_data, round):
    '''生成m2i所需格式'''
    inputs, decoded_example = _parse(input_data)
    sample_is_valid = inputs['sample_is_valid']
    tracks_to_predict = tf.boolean_mask(inputs['tracks_to_predict'], sample_is_valid)
    predict_agent_num = tracks_to_predict.numpy().sum()
    tracks_type = tf.boolean_mask(decoded_example['state/type'], sample_is_valid)
    tracks_type = tracks_type.numpy().copy().reshape(-1)
    instance = []
    for select in range(predict_agent_num):
        # Make sure both agents are of the same type if it is specified.
        # if type_is_ok(tracks_type[select], args) and type_is_ok(tracks_type[1 - select], args):
        if True:
            t = get_instance(args, inputs, decoded_example, str(decoded_example['influencer_reactor_idx'][0])+'_'+str(decoded_example['influencer_reactor_idx'][1])+'_'+str(round),
                                select=select)
            # change to a soft check when eval
            if t is not None:
                instance.append(t)

    assert len(instance) == 1, len(instance)
    return instance

def get_metric_params(mapping, args):
    gt_trajectory = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'gt_trajectory'))
    gt_is_valid = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'gt_is_valid'))
    object_type = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'object_type'))
    scenario_id = utils.get_from_mapping(mapping, 'scenario_id')[0]
    object_id = tf.convert_to_tensor(utils.get_from_mapping(mapping, 'object_id'))

    if 'joint_eval' in args.other_params:
        assert len(mapping) == 2, len(mapping)
        gt_trajectory = gt_trajectory[tf.newaxis, :]
        gt_is_valid = gt_is_valid[tf.newaxis, :]
        object_type = object_type[tf.newaxis, :]

    return (
        gt_trajectory,
        gt_is_valid,
        object_type,
        scenario_id,
        object_id
    )

def get_traj_nearest_to_GT(gt_trajectory_x, gt_trajectory_y, pred_trajectory, k, v, distance_type='chamfer'):
    '''找到和gt最近的预测轨迹'''
    valid_gt_trajectory_x = np.array(gt_trajectory_x[gt_trajectory_x!=-1]).reshape(-1,1)
    valid_gt_trajectory_y = np.array(gt_trajectory_y[gt_trajectory_y!=-1]).reshape(-1,1)
    assert valid_gt_trajectory_x.shape == valid_gt_trajectory_y.shape
    if valid_gt_trajectory_x.shape[0] == 0: # gt disappear
        return pred_trajectory[0,:,0], pred_trajectory[0,:,1]
    assert valid_gt_trajectory_x.shape[0] != 0
    gt_trajectory = np.concatenate([valid_gt_trajectory_x, valid_gt_trajectory_y], axis=1)

    if distance_type == 'chamfer':
        # 1
        neigh_1 = NearestNeighbors(n_neighbors=1)
        neigh_1.fit(gt_trajectory)

        distance_list_1 = np.zeros(pred_trajectory.shape[0])
        for i in range(pred_trajectory.shape[0]):
            dist, indexes = neigh_1.kneighbors(pred_trajectory[i][:valid_gt_trajectory_x.shape[0]], return_distance=True)
            distance_list_1[i] = dist.mean()

        # 2
        distance_list_2 = np.zeros(pred_trajectory.shape[0])
        for i in range(pred_trajectory.shape[0]):
            neigh_2 = NearestNeighbors(n_neighbors=1)
            neigh_2.fit(pred_trajectory[i][:valid_gt_trajectory_x.shape[0]])

            dist, indexes = neigh_2.kneighbors(gt_trajectory, return_distance=True)
            distance_list_2[i] = dist.mean()

        # distance_list = distance_list_1 + distance_list_2
        distance_list = distance_list_2
        # return
        # if k == 0:
        #     embed()
        min_idx = np.argmin(distance_list)
    elif distance_type == 'per_point':
        distance_list = np.zeros(pred_trajectory.shape[0])
        for i in range(pred_trajectory.shape[0]):
            dist = np.sqrt(((gt_trajectory[:valid_gt_trajectory_x.shape[0]] - pred_trajectory[i][:valid_gt_trajectory_x.shape[0]]) ** 2).sum(1))
            final_dist = min(np.sqrt(((gt_trajectory[valid_gt_trajectory_x.shape[0]-1] - pred_trajectory[i][:valid_gt_trajectory_x.shape[0]]) ** 2).sum(1)))

            distance_list[i] = dist.mean() + final_dist*0.3

        min_idx = np.argmin(distance_list)
        # if k == 0:
        #     embed()
    else:
        raise NotImplementedError

    #print(min_idx)
    # embed(header='get_nearest_gt')
    return pred_trajectory[min_idx,:,0], pred_trajectory[min_idx,:,1]

def get_traj_nearest_to_lane(road_xyz, pred_trajectory):
    '''找到和车道线最近的轨迹'''
    road_xyz_array = np.array(road_xyz)[:,:2]

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(road_xyz_array)

    distance_list = np.zeros(pred_trajectory.shape[0])
    for i in range(pred_trajectory.shape[0]):
        dist, indexes = neigh.kneighbors(pred_trajectory[i], return_distance=True)
        distance_list[i] = dist.mean()

    min_idx = np.argmin(distance_list)

    return pred_trajectory[min_idx,:,0], pred_trajectory[min_idx,:,1]

def get_inter_relation(gt_trajectory_x, gt_trajectory_y, influencer_gt_x, influencer_gt_y):
    '''判断从车轨迹是否需要预测，需要返回True，否则False'''
    distance_threshold = 5

    valid_gt_trajectory_x = np.array(gt_trajectory_x[gt_trajectory_x!=-1])
    valid_gt_trajectory_y = np.array(gt_trajectory_y[gt_trajectory_y!=-1])
    assert valid_gt_trajectory_x.shape == valid_gt_trajectory_y.shape
    if valid_gt_trajectory_x.shape[0] < influencer_gt_x.shape[0]: # gt disappear
        return True

    distance = np.sqrt((valid_gt_trajectory_x.reshape(-1,1) - influencer_gt_x)**2 + \
                        (valid_gt_trajectory_y.reshape(-1,1) - influencer_gt_y)**2)
    d_min = distance.min()
    if d_min < distance_threshold:
        return True
    else:
        return False

round = 0
def run_m2i_inference(args, m2i_model, input_data, pred_traj_type='highestScore', filter_noninter=False):
    global round

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    import tensorflow as tf
    global tf

    eval_prediction = False
    if eval_prediction:
        from .waymo_tutorial import MotionMetrics, metrics_config, metric_names
        motion_metrics = MotionMetrics(metrics_config)
        if args.mode_num != 6:
            motion_metrics.not_compute = True
        utils.motion_metrics = MotionMetrics(metrics_config)
        utils.metric_names = metric_names

    split_data_list, influencer_idx = choose_reactor(input_data, args)

    predicted_traj_x_highestScore = np.ones(input_data['state/future/x'].shape) * -1.
    predicted_traj_x_highestScore[influencer_idx] = copy.deepcopy(input_data['state/future/x'])[influencer_idx]
    predicted_traj_y_highestScore = np.ones(input_data['state/future/y'].shape) * -1.
    predicted_traj_y_highestScore[influencer_idx] = copy.deepcopy(input_data['state/future/y'])[influencer_idx]

    predicted_traj_x_nearestGT = np.ones(input_data['state/future/x'].shape) * -1.
    predicted_traj_x_nearestGT[influencer_idx] = copy.deepcopy(input_data['state/future/x'])[influencer_idx]
    predicted_traj_y_nearestGT = np.ones(input_data['state/future/y'].shape) * -1.
    predicted_traj_y_nearestGT[influencer_idx] = copy.deepcopy(input_data['state/future/y'])[influencer_idx]

    predicted_traj_x_nearestLane = np.ones(input_data['state/future/x'].shape) * -1.
    predicted_traj_x_nearestLane[influencer_idx] = copy.deepcopy(input_data['state/future/x'])[influencer_idx]
    predicted_traj_y_nearestLane = np.ones(input_data['state/future/y'].shape) * -1.
    predicted_traj_y_nearestLane[influencer_idx] = copy.deepcopy(input_data['state/future/y'])[influencer_idx]

    all_traj = np.ones([input_data['state/future/x'].shape[0],6,80,2]) * -1.

    for k,v in split_data_list.items():
        modified_data_item = prepare_data(args, v, round)

        if eval_prediction:
            metric_params = get_metric_params(modified_data_item, args)

        if filter_noninter: # TODO 若某一轮进行了预测，则后续轮均需进行预测
            inter_or_not = get_inter_relation(v['dummy/future/gt/x'][1], v['dummy/future/gt/y'][1], \
                                                input_data['state/future/x'][influencer_idx], input_data['state/future/y'][influencer_idx])
            if inter_or_not == False:
                predicted_traj_x_highestScore[k] = np.array(v['dummy/future/gt/x'][1])
                predicted_traj_y_highestScore[k] = np.array(v['dummy/future/gt/y'][1])
                predicted_traj_x_nearestGT[k]    = np.array(v['dummy/future/gt/x'][1])
                predicted_traj_y_nearestGT[k]    = np.array(v['dummy/future/gt/y'][1])
                predicted_traj_x_nearestLane[k]  = np.array(v['dummy/future/gt/x'][1])
                predicted_traj_y_nearestLane[k]  = np.array(v['dummy/future/gt/y'][1])
                continue

        pred_trajectory, pred_score, _ = m2i_model(modified_data_item, device)
        all_traj[k] = copy.deepcopy(pred_trajectory[0])

        if eval_prediction:
            motion_metrics.args = args
            motion_metrics.update_state(
                tf.convert_to_tensor(pred_trajectory.astype(np.float32)),
                tf.convert_to_tensor(pred_score.astype(np.float32)),
                *metric_params
            )

        if pred_traj_type == 'highestScore':
            predicted_traj_x_highestScore[k] = copy.deepcopy(pred_trajectory[0,0,:,0])  # batch, highest score, time, x
            predicted_traj_y_highestScore[k] = copy.deepcopy(pred_trajectory[0,0,:,1])
        elif pred_traj_type == 'nearestGT':
            #print('========k:', k)
            predicted_traj_x_nearestGT[k], predicted_traj_y_nearestGT[k] = get_traj_nearest_to_GT(v['dummy/future/gt/x'][1], v['dummy/future/gt/y'][1], pred_trajectory[0,:,:,:], k, v)
        elif pred_traj_type == 'nearestLane':
            predicted_traj_x_nearestLane[k], predicted_traj_y_nearestLane[k] = get_traj_nearest_to_lane(v['roadgraph_samples/xyz'], pred_trajectory[0,:,:,:])
        else:
            raise NotImplementedError
        
        # print(k)

    if eval_prediction:
        print('all metric_values', len(motion_metrics.get_all()[0]))
        print(utils.metric_values_to_string(motion_metrics.result(), metric_names))

    round += 1
    if pred_traj_type == 'highestScore':
        return predicted_traj_x_highestScore, predicted_traj_y_highestScore
    elif pred_traj_type == 'nearestGT':
        return predicted_traj_x_nearestGT, predicted_traj_y_nearestGT
    elif pred_traj_type == 'nearestLane':
        return predicted_traj_x_nearestLane, predicted_traj_y_nearestLane
    else:
        raise NotImplementedError

def main():
    '''test'''
    ckpt_path = '/DATA2/lpf/baidu/additional_files_for_m2i/test0922_0_vv/model_save/model.12.bin'
    reactor_type = 'vehicle'
    args, m2i_model = init_m2i(ckpt_path, reactor_type)

    input_data = np.load('/DATA2/lpf/baidu/M2I-main/trajactory/test.npy', allow_pickle=True).item()
    input_data.pop('relation')

    redicted_traj_x, predicted_traj_y = run_m2i_inference(args, m2i_model, input_data, pred_traj_type='nearestGT')

    embed(header='main')

if __name__ == '__main__':
    main()

