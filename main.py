import torch
torch.cuda.set_device(0)


USE_M2I = True
USE_MPC = False


import os
MANUAL_SCENEE_NAME = '14_2'
SUFFIX = 'yihzuang#14_2__all__right_line'
print('Scene name is:', MANUAL_SCENEE_NAME)


MANUAL_SDC_ID = -1
print('Self-driving car ID is:', MANUAL_SDC_ID)


MANUAL_TH = 2.0
if MANUAL_TH != 0:
    print('Threshold for noise suppression is:', MANUAL_TH)
if (MANUAL_TH < 0.5 or MANUAL_TH > 3.0) and MANUAL_TH !=0:
    print('Recommended range for noise suppression threshold is between 0.5 and 3.0')
if MANUAL_TH == 0:
    print('Disabled noise suppression')


MANUAL_VIS_TH = 1.0
print('Threshold for visualization yaw noise suppression is:', MANUAL_VIS_TH)


MANUAL_MAX_DIFFERENCE = 90
print('Threshold for U-turn detection is:', MANUAL_MAX_DIFFERENCE)


MANUAL_PREDICTION_FREQUENCY = 7.5
MANUAL_NUM_FUTURE_TO_DISCARD = 80 - int(MANUAL_PREDICTION_FREQUENCY * 10)
print('Update future trajectory every', (80 - MANUAL_NUM_FUTURE_TO_DISCARD) / 10, 'seconds')
if MANUAL_PREDICTION_FREQUENCY <= 0 or MANUAL_PREDICTION_FREQUENCY >= 8:
    print('Invalid frequency.')
    raise ValueError()


DEBUG = False
EXPRESS_VIS = False
if (DEBUG == False) and (USE_MPC == False and USE_M2I == False):
    MANUAL_TH = 0
    EXPRESS_VIS = True


from loguru import logger 
import pickle5 as pickle
import pandas as pd
import numpy as np
import json
import copy
import matplotlib.pyplot as plt
import sys
import warnings 
warnings.filterwarnings('ignore')
sys.path.append('src')


path_whole = '/DATA1/liyang/M2I_2/data_split_new/' + MANUAL_SCENEE_NAME + '/' + SUFFIX + '.pickle'
path = '/DATA1/liyang/M2I_2/data_split_new/' + MANUAL_SCENEE_NAME + '_split/' + SUFFIX + '.pickle'
with open(path, 'rb') as f:
    decoded_example_group = pickle.load(f)
decoded_example_group = decoded_example_group
with open(path_whole, 'rb') as f:
    decoded_whole = pickle.load(f)


naive_traffic_light_keys = []
for key in decoded_example_group.keys():
    if key.find('traffic_light') > -1:
        naive_traffic_light_keys.append(key)
for naive_traffic_light_key in naive_traffic_light_keys:
    slash_index = naive_traffic_light_key.find('/')
    prefix = naive_traffic_light_key[:slash_index]
    suffix = naive_traffic_light_key[slash_index+1:]
    traffic_current_key = prefix + '/' + 'current' + '/' + suffix
    traffic_future_key = prefix + '/' + 'future' + '/' + suffix
    decoded_example_group[traffic_future_key] = decoded_example_group[naive_traffic_light_key]
    decoded_example_group[traffic_current_key] = decoded_example_group[naive_traffic_light_key][:,0:1]

if MANUAL_SDC_ID == -1:
    sdc_id = int(decoded_example_group['state/id'][np.where(decoded_example_group['state/is_sdc'])[0][0]])
    MANUAL_SDC_ID = sdc_id
else:
    sdc_id = MANUAL_SDC_ID
print('sdc_id is:', sdc_id)


CURRENT_INDEX_IN_WHOLE = 10
START_INDEX_IN_WHOLE = 10


naive_keys = ['x', 'y', 'length', 'width', 'bbox_yaw', 'vel_yaw', 'velocity_x', 'velocity_y']
prefix_past = 'state/past'
prefix_future = 'state/future'
prefix_current = 'state/current'
desired_keys_past = [prefix_past + '/' + key for key in naive_keys] 
desired_keys_future = [prefix_future + '/' + key for key in naive_keys] 
desired_keys_current = [prefix_current + '/' + key for key in naive_keys] 
all_column_names_for_df = desired_keys_past + desired_keys_current + desired_keys_future


dummy_past = filter_dict_by_keys(decoded_example_group, ['state/id'] + desired_keys_past, start_index=0, end_index=4)
dummy_future = filter_dict_by_keys(decoded_example_group, ['state/id'] + desired_keys_future, start_index=2, end_index=7)
dummy_current = filter_dict_by_keys(decoded_example_group, ['state/id'] + desired_keys_current, start_index=0, end_index=4)
dummy_current_2 = filter_dict_by_keys(decoded_example_group, ['state/id'] + desired_keys_current, start_index=2, end_index=11)


roadgraph_dict = {}
for key in decoded_example_group.keys():
    if key.find('roadgraph_samples') >= 0:
        roadgraph_dict[key] = decoded_example_group[key]     
traffic_light_dict = {}
for key in decoded_example_group.keys():
    if key.find('traffic_light') >= 0:
        traffic_light_dict[key] = decoded_example_group[key]

past = filter_dict_by_keys(decoded_example_group, ['state/id'] + desired_keys_past)
current = filter_dict_by_keys(decoded_example_group, ['state/id'] + desired_keys_current)
df = create_and_fill_df_for_ids(all_column_names_for_df, past_dict_to_use=past, current_dict_to_use=current, future_dict_to_use=None)


# Drop invalid rows
df = df[df['state/past/x'].apply(check_valid_past)]
df = df.reset_index(drop=True)

# M2I
if USE_M2I:
    from src.m2i_script import *
    m2i_checkpoint_path = '/DATA2/lpf/baidu/additional_files_for_m2i/test0908_0/model_save/model.12.bin'
    args, m2i_model = init_m2i(m2i_checkpoint_path, reactor_type = 'vehicle')
# MPC
if USE_MPC:
    from src.m2i_mpc_ultra import *

#######################################
# Debug Information

mpc_output_vel_yaw_list = []
mpc_output_bbox_yaw_list = []
mpc_valid_index_list = []

dicts_to_update_df = []
list_for_mpc_valid_indices = []
list_for_raw_mpc_outputs = []
list_for_sample_mpc_inputs = []
list_for_filtered_mpc_inputs = []

list_for_res = []
#######################################

for i in range(999999):
    
    
    prefix = 'Round ' + str(i) + ': '
    update_stdout(prefix)
    
    # Get inputs for m2i
    try:
        res = get_inputs_for_M2I(df, sdc_id)
    except:
        print('Break when preparing outputs.')
        break
    
    
    # Prepare velocity and yaw
    if i == 0: 
        # for the first round, use ground truth yaw
        res = use_gt_past_and_current_velocity_yaw_for_the_first_round(res)
        res = velocity_yaw_future(res)
    else:
        # otherwise, calculate vel and yaw and remove oscilations
        # 1. update vel and yaw before running m2i
        res = velocity_yaw_future(velocity_yaw_current(velocity_yaw_past(res)))
        # 2. remove oscilations in past and current (x and y based on vel x and vel y)
        res = noise_suppression(res, 'past', th=MANUAL_TH)
    
    
    
    # find and add gt trajectory for all cars as dummy
    res, gt_traj_x, gt_traj_y = add_gt_future_trajectory_as_dummy(res)
    
    
    
    ###############################################
    # Debug
    list_for_res.append(res)
    ###############################################
    
    # [3. Run M2I]
    if USE_M2I:
        if (res['state/past/x'] == -1).all():
            print('Break from M2I. M2I input contains no valid trajectory.')
            break
        trajectory_type = 'nearestGT' # (highestScore, nearestGT, nearestLane)       
        try:
            predicted_traj_x, predicted_traj_y = run_m2i_inference(args, m2i_model, res, trajectory_type) 
        except:
            print('Break from M2I through except. TODO: check M2I inputs before calling M2I.')
            break
        
    else:
        predicted_traj_x, predicted_traj_y = gt_traj_x, gt_traj_y
        
    # [4. Run MPC]
    # 4.1 use M2I outputs to updated future trajectories
    sample_mpc_inputs = get_inputs_for_mpc(res, predicted_traj_x, predicted_traj_y)
    
    
    index_mapping = find_indices_in_whole(df, decoded_whole)

    sample_mpc_inputs['state/dummy/complete_gt/x'] = decoded_whole['state/x'][index_mapping]
    sample_mpc_inputs['state/dummy/complete_gt/y'] = decoded_whole['state/y'][index_mapping]
    sample_mpc_inputs['state/dummy/complete_gt/velocity_x'] = decoded_whole['state/velocity_x'][index_mapping]
    sample_mpc_inputs['state/dummy/complete_gt/velocity_y'] = decoded_whole['state/velocity_y'][index_mapping]
    sample_mpc_inputs['state/dummy/complete_gt/bbox_yaw'] = decoded_whole['state/bbox_yaw'][index_mapping]
    sample_mpc_inputs['state/dummy/complete_gt/vel_yaw'] = decoded_whole['state/vel_yaw'][index_mapping]
    
    
    
    ######################################################################
    # Debug
    list_for_sample_mpc_inputs.append(sample_mpc_inputs)
    ######################################################################
    
    if USE_MPC:
        # 4.2 speed up MPC inference by removing the second half of each trajectory
        shortened_mpc_inputs = shorten_future_in_dict(sample_mpc_inputs, desired_future_length=80-MANUAL_NUM_FUTURE_TO_DISCARD+1)
        # 4.3 filter out trajectories that cannot be processed by MPC
        filtered_mpc_inputs, mpc_valid_indices = filter_inputs_for_mpc(shortened_mpc_inputs, th=MANUAL_TH, max_difference=MANUAL_MAX_DIFFERENCE)
        # 4.4 calculate future velocity for MPC
        filtered_mpc_inputs = velocity_yaw_future(filtered_mpc_inputs)
        # (Break if mpc inputs contains no trajectory)
        if filtered_mpc_inputs['state/future/x'].shape[0] == 0:
            print('Break from MPC. MPC input contains no trajectory.')
            break
            
        ######################################################################    
        list_for_filtered_mpc_inputs.append(filtered_mpc_inputs)
        ######################################################################
        
        
        # 4.5 call MPC
        mpc_outputs = mpc_forward(filtered_mpc_inputs)
        
        ######################################################################
        # Debug
        mpc_output_vel_yaw_list.append(mpc_outputs['state/future/vel_yaw'])
        mpc_output_bbox_yaw_list.append(mpc_outputs['state/future/bbox_yaw'])
        mpc_valid_index_list.append(mpc_valid_indices)
        
        list_for_raw_mpc_outputs.append(mpc_outputs)
        ######################################################################
        # 4.6 Remove dummy and update MPC inputs (with all trajectories) using MPC outputs (with only filtered trajectories)
        mpc_outputs =  process_mpc_outputs(sample_mpc_inputs, mpc_outputs, mpc_valid_indices)
    else:
        mpc_outputs = copy.deepcopy(sample_mpc_inputs)
    
    ######################################################################
    # Debug
    dicts_to_update_df.append(mpc_outputs)
    if USE_MPC:
        list_for_mpc_valid_indices.append(mpc_valid_indices)
        
    ######################################################################
    
    # [5. Update df]
    df = update_df_using_MPC_outputs(df, mpc_outputs, num_future_samples_to_use_as_future=MANUAL_NUM_FUTURE_TO_DISCARD)


def progressBar(i, max, text):
    bar_size = 60
    j = (i + 1) / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()
    
def update_stdout(text_to_display):
    sys.stdout.write('\r')
    sys.stdout.write(text_to_display)
    sys.stdout.flush()

def isNaN(num):
    res = num != num
    if type(res) != bool:
        return res.all()
    else:
        return res

def filter_dict_by_keys(old_dict: dict, desired_keys: list, start_index = None, end_index = None):
    if start_index is None and end_index is None:
        return {k:old_dict[k] for k in desired_keys}
    else:
        return {k:old_dict[k][start_index:end_index] for k in desired_keys}
    
def list_to_float(x):
    if type(x) == list:
        return x[0]
    else:
        return x

def velocity_yaw_future(old_decoded_example):
    
    """
    For future trajectories, calculate velocity, velocity yaw, 
    and bbox yaw using x and y values. The values of bbox yaw
    are set to be the same as velocity yaw. 
    
    Parameters
    ----------
    old_decoded_example: dict
        Input for M2I. 
        
    
    Returns
    -------
    decoded_example: dict
        Input for M2I with updated velocity, velocity yaw, 
        and bbox_yaw for future trajectories. 
    
    """
    
    
    
    decoded_example = copy.deepcopy(old_decoded_example)
    
    x = decoded_example['state/future/x']
    y = decoded_example['state/future/y']
    velocity_x = np.zeros((x.shape[0], x.shape[1]))
    velocity_y = np.zeros((x.shape[0], x.shape[1]))
    vel_yaw = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]-1):
        velocity_x[:,i] = 10*(x[:,i+1]-x[:,i])
        velocity_y[:,i] = 10*(y[:,i+1]-y[:,i])
    velocity_x[:,-1] = velocity_x[:,-2]
    velocity_y[:,-1] = velocity_y[:,-2]
    vel_yaw = np.arctan2(velocity_y, velocity_x)
    
    decoded_example['state/future/velocity_x'] = velocity_x
    decoded_example['state/future/velocity_y'] = velocity_y
    decoded_example['state/future/vel_yaw'] = vel_yaw
    decoded_example['state/future/bbox_yaw'] = vel_yaw
    
    for key in decoded_example:
        try:
            if decoded_example[key].dtype == np.float64:
                decoded_example[key] = np.float32(decoded_example[key])
        except:
            pass

    return decoded_example

def velocity_yaw_past(old_decoded_example):
    
    """
    For past trajectories, calculate velocity, velocity yaw, 
    and bbox yaw using x and y values. The values of bbox yaw
    are set to be the same as velocity yaw. 
    
    Parameters
    ----------
    old_decoded_example: dict
        Input for M2I. 
        
    
    Returns
    -------
    decoded_example: dict
        Input for M2I with updated velocity, velocity yaw, 
        and bbox_yaw for past trajectories. 
    
    """
    
    
    decoded_example = copy.deepcopy(old_decoded_example)
    
    x = decoded_example['state/past/x']
    y = decoded_example['state/past/y']
    velocity_x = np.zeros((x.shape[0], x.shape[1]))
    velocity_y = np.zeros((x.shape[0], x.shape[1]))
    vel_yaw = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]-1):
        velocity_x[:,i] = 10*(x[:,i+1]-x[:,i])
        velocity_y[:,i] = 10*(y[:,i+1]-y[:,i])
    
    # # Handle near static cars
    # for car in range(velocity_x.shape[0]):
    #     for time in range(velocity_x.shape[1]):
    #         if velocity_x
    
    
    velocity_x[:,-1] = velocity_x[:,-2]
    velocity_y[:,-1] = velocity_y[:,-2]
    vel_yaw = np.arctan2(velocity_y, velocity_x)
    
    decoded_example['state/past/velocity_x'] = velocity_x
    decoded_example['state/past/velocity_y'] = velocity_y
    decoded_example['state/past/vel_yaw'] = vel_yaw
    decoded_example['state/past/bbox_yaw'] = vel_yaw

    for key in decoded_example:
        try:
            if decoded_example[key].dtype == np.float64:
                decoded_example[key] = np.float32(decoded_example[key])
        except:
            pass
        
    return decoded_example

def velocity_yaw_current(old_decoded_example):
    
    """
    For current trajectories, set velocity, velocity yaw, 
    and bbox yaw. Their values are set to be the same as 
    the latest values in past trajectories. 
    
    Parameters
    ----------
    old_decoded_example: dict
        Input for M2I. 
        
    
    Returns
    -------
    decoded_example: dict
        Input for M2I with updated velocity, velocity yaw, 
        and bbox_yaw for current trajectories. 
    
    """
    
    decoded_example = copy.deepcopy(old_decoded_example)
    
    decoded_example['state/current/velocity_x'] = decoded_example['state/past/velocity_x'][:,-1:]
    decoded_example['state/current/velocity_y'] = decoded_example['state/past/velocity_y'][:,-1:]
    decoded_example['state/current/bbox_yaw'] = decoded_example['state/past/bbox_yaw'][:,-1:]
    decoded_example['state/current/vel_yaw'] = decoded_example['state/past/vel_yaw'][:,-1:]
    
    return decoded_example

def noise_suppression(dict_to_suppress, split, th=2.5):
    
    res = copy.deepcopy(dict_to_suppress)
    
    for car_index in range(len(res['state/id'])):
        x = res['state/' + split + '/x'][car_index]
        y = res['state/' + split + '/y'][car_index]
        vel_x = res['state/' + split + '/velocity_x'][car_index]
        vel_y = res['state/' + split + '/velocity_y'][car_index]
        vel_yaw = np.arctan2(vel_y, vel_x)
        
        threashold = th
        mask = (np.abs(vel_x) < threashold) * (np.abs(vel_y) < threashold)
        for index in range(len(mask)):
            current_x = x[index]
            current_y = y[index]

            if mask[index]:
                found = False
                for previous_index in range(index - 1, -1, -1):
                    if not mask[previous_index]:
                        x[index] = x[previous_index]
                        y[index] = y[previous_index]
                        vel_yaw[index] = np.arctan2(vel_y[previous_index], vel_x[previous_index])
                        vel_x[index] = 0
                        vel_y[index] = 0
                        
                        #mask[index] = False
                        found = True
                        break
                if not found:
                    # if nothing in the past was greater than threshold: use the first one
                    x[index] = x[0]
                    y[index] = y[0]
                    vel_yaw[index] = vel_yaw[0]
                    vel_x[index] = 0
                    vel_y[index] = 0
        
        res['state/' + split + '/x'][car_index] = x
        res['state/' + split + '/y'][car_index] = y
        res['state/' + split + '/vel_yaw'][car_index] = vel_yaw
        res['state/' + split + '/bbox_yaw'][car_index] = vel_yaw
    
    # modify current if split is past
    if split == 'past':
        res['state/current/velocity_x'] = res['state/past/velocity_x'][:,-1:]
        res['state/current/velocity_y'] = res['state/past/velocity_y'][:,-1:]
        res['state/current/bbox_yaw'] = res['state/past/bbox_yaw'][:,-1:]
        res['state/current/vel_yaw'] = res['state/past/vel_yaw'][:,-1:]
        
    return res

def calculate_interval(input_angle, max_difference=90):
    
    interval_list = []
    
    low = input_angle - max_difference
    high = input_angle + max_difference
    
    if low <= -180:
        interval_list.append([-180, input_angle])
        extra_low = low + 360
        extra_high = 180
        interval_list.append([extra_low, extra_high])
    else:
        interval_list.append([low, input_angle])
        
        
    if high >= 180:
        interval_list.append([input_angle, 180])
        extra_low = -180
        extra_high = high - 360
        interval_list.append([extra_low, extra_high])
    else:
        interval_list.append([input_angle, high])
        
    return interval_list

def yaw_to_degree(yaw):
    degree = yaw * 180 / np.pi
    return degree

def detect_u_turn_in_one_trajectory(x, y, max_difference=90):
    
    assert len(x.shape) == 1 and len(y.shape) == 1
    
    exist_u_turn = False
        
    x = np.expand_dims(x,0)
    y = np.expand_dims(y,0)   
    velocity_x = np.zeros((x.shape[0], x.shape[1]))
    velocity_y = np.zeros((x.shape[0], x.shape[1]))
    vel_yaw = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]-1):
        velocity_x[:,i] = 10*(x[:,i+1]-x[:,i])
        velocity_y[:,i] = 10*(y[:,i+1]-y[:,i])
    vel_yaw = np.arctan2(velocity_y, velocity_x)
    vel_yaw = vel_yaw[0]
    vel_yaw[-1] = vel_yaw[-2]
        
    yaw_in_degree = yaw_to_degree(vel_yaw)
    for current_angle in yaw_in_degree:
        valid_interval_list = calculate_interval(current_angle, max_difference=max_difference)
        if current_angle == 0:
            continue
        for another_angle in yaw_in_degree:
            if another_angle == 0:
                continue
            
            in_at_least_one_interval = False
            
            for valid_interval in valid_interval_list:
                if another_angle >= valid_interval[0] and another_angle <= valid_interval[1]:
                    in_at_least_one_interval = True 
                    break
            if not in_at_least_one_interval: 
                exist_u_turn = True
                break
                
        if exist_u_turn:
            break
            
    return exist_u_turn

def find_u_turns(m2i_outputs, th=2.5, max_difference=45):
    res = copy.deepcopy(m2i_outputs)
    
    # calculate velocity and yaw for the predicted future trajactories
    res = velocity_yaw_future(res)
    # noise suppression for predicted future trajectories
    res = noise_suppression(res, 'future', th=th)
    
    # create an empty list to indicate which trajectories contain u-turns
    contain_u_turns = []
    
    # detect u turns for each predicted future trajectory
    for index in range(res['state/id'].shape[0]):
        x = res['state/future/x'][index]
        y = res['state/future/y'][index]
        current_trajectory_has_u_turn = detect_u_turn_in_one_trajectory(x, y, max_difference=max_difference)
        contain_u_turns.append(False)
    
    return np.array(contain_u_turns)

def filter_dict_using_indices(input_dict, indices):
    
    # indices is an array of boolean values
    
    filtered_inputs = {}

    for key in input_dict.keys():
        if type(input_dict[key]) == str:
            filtered_inputs[key] = input_dict[key]
        elif 'traffic_light' in key:
            filtered_inputs[key] = input_dict[key]
        elif 'roadgraph_samples' in key:
            filtered_inputs[key] = input_dict[key]
        else:
            filtered_inputs[key] = input_dict[key][indices]
            
    return filtered_inputs

def update_dict_using_indices(original_dict, new_dict, indices):
    
    # indices is an array of boolean values
    
    updated_dict = {}
    
    for key in new_dict.keys():
        if type(new_dict[key]) == str or 'traffic_light' in key or 'roadgraph_samples' in key:
            updated_dict[key] = new_dict[key]
        else:
            if len(original_dict[key].shape) == 2 and 'future' in key:
                
                # Fix (MPC) outputs of incorrect shape
                if len(new_dict[key].shape) == 1:
                    new_dict[key] = np.float32(np.array([new_dict[key]]))
                
                # Number of samples for each trajectory in MPC outputs
                num_samples = new_dict[key].shape[1]

                # Update original dict (MPC inputs) using new dict (MPC outputs)
                #######################################################
                # Unable to assign value to numpy 2D array on the LHS
                # This is an workaround
                
                # Copy left
                left_copy = []
                for row in range(original_dict[key].shape[0]):
                    current_row = []
                    for col in range(original_dict[key].shape[1]):
                        current_row.append(original_dict[key][row, col])
                    left_copy.append(current_row)

                # update left
                new_dict_row_index = 0
                for row in range(len(left_copy)):
                    need_to_update_current_row = indices[row]
                    if need_to_update_current_row:
                        for col in range(num_samples):
                            left_copy[row][col] = new_dict[key][new_dict_row_index, col]
                        new_dict_row_index += 1

                # list of lists to a 2D array
                inner_as_array = [np.array(row) for row in left_copy]
                outer_as_array = np.array(inner_as_array)

                # To float32
                outer_as_array = np.float32(outer_as_array)

                #######################################################
                
                updated_dict[key] = outer_as_array

        
            else: 
                # updated_dict[key][indices] = new_dict[key]
                updated_dict[key] = original_dict[key]

            
    return updated_dict

def rank_distance(filtered_mpc_inputs):
    distance_list = []

    for trajectory_index in range(filtered_mpc_inputs['state/id'].shape[0]):
        all_x = filtered_mpc_inputs['state/future/x'][trajectory_index]
        all_y = filtered_mpc_inputs['state/future/y'][trajectory_index]

        trajectory_distance = 0.0

        for point_index in range(1, all_x.shape[0]):

            x = all_x[point_index]
            y = all_y[point_index]

            previous_x = all_x[point_index - 1]
            previous_y = all_y[point_index - 1]

            # Calculate distance
            x_distance_square = np.square(x - previous_x)
            y_distance_square = np.square(y - previous_y)

            step_distance = np.sqrt(x_distance_square + y_distance_square)
            trajectory_distance += step_distance

        distance_list.append(trajectory_distance)

    distance_list = np.array(distance_list)

    order_descending = np.argsort(-distance_list)
    
    return order_descending

def delete_dummy_from_dict(input_dict):
    keys_to_delete = []
    for key in input_dict.keys():
        if 'dummy' in key:
            keys_to_delete.append(key)
    if len(keys_to_delete) > 0:
        for key in keys_to_delete:
            del input_dict[key]
    return input_dict

def fill_empty_df_using_dict(empty_df, past_dict_to_use=None, current_dict_to_use=None, future_dict_to_use=None):
    if past_dict_to_use is not None:
        for name in past_dict_to_use.keys():
            if name == 'state/id':
                continue
            empty_df[name] = past_dict_to_use[name].tolist()
    if current_dict_to_use is not None:
        for name in current_dict_to_use.keys():
            if name == 'state/id':
                continue
            empty_df[name] = current_dict_to_use[name]
    if future_dict_to_use is not None:
        for name in future_dict_to_use.keys():
            if name == 'state/id':
                continue
            empty_df[name] = future_dict_to_use[name].tolist()
    return empty_df

def create_and_fill_df_for_ids(all_column_names_for_df, past_dict_to_use=None, current_dict_to_use=None, future_dict_to_use=None, ids=None):
    
    if ids is not None:
        pass
    elif past_dict_to_use is not None:
        ids = past_dict_to_use['state/id']
    elif current_dict_to_use is not None:
        ids = current_dict_to_use['state/id']
    elif future_dict_to_use is not None:
        ids = future_dict_to_use['state/id']
    
    new_df = pd.DataFrame({'state/id':ids})
    for name in all_column_names_for_df:
        new_df[name] = np.nan
    new_df = fill_empty_df_using_dict(new_df, past_dict_to_use=past_dict_to_use, current_dict_to_use=current_dict_to_use, future_dict_to_use=future_dict_to_use)
    
    # Transform lists (with is expected to contain only one element) to float
    for desired_key in desired_keys_current:
        new_df[desired_key] = new_df[desired_key].apply(list_to_float)
        
    return new_df

def check_invalid_past(x):
    return (pd.Series(x) == -1).all()
    
def check_valid_past(x):
    return (pd.Series(x) != -1).any()

def find_index_of_id_in_df(df, object_id):
    mask = df['state/id'] == object_id
    id_already_in_df = mask.any()
    if id_already_in_df:
        index = df[mask].index[0]
        return index
    else:
        return -1

def update_cell(df, row_index_in_df, column_to_update, new_cell_data, replace_items=False):
    
    # update cell if id already in df
    if row_index_in_df >= 0:
        # Updating cells whose values are lists (past and future) by contatenating
        if not replace_items:
            if not isNaN(df[column_to_update].iloc[row_index_in_df]):
                right_half = new_cell_data.tolist()
                # when using current (float) to update past (list), change float to list before concatenating them
                if not type(right_half) == list:
                    right_half = [right_half]
                df.at[row_index_in_df, column_to_update] = df[column_to_update].iloc[row_index_in_df] + right_half
            else:
                try:
                    df.at[row_index_in_df, column_to_update] = new_cell_data.tolist()
                except:
                    df[column_to_update] = df[column_to_update].astype(object)
                    df.at[row_index_in_df, column_to_update] = new_cell_data.tolist()
        # Updating cells whose values are not floats (current) by overwriting
        else:
            if not isNaN(df[column_to_update].iloc[row_index_in_df]):
                df.at[row_index_in_df, column_to_update] = new_cell_data
            else:
                df[column_to_update] = df[column_to_update].astype(object)
                df.at[row_index_in_df, column_to_update] = new_cell_data

    # does not work if id is not in df
    else:
        raise ValueError("row_index_in_df should be non-negative (id should already be in the df)")
        
def update_row(df, row_index_in_df, columns_to_update, columns_of_new_data, new_data, index_in_new_data, replace_items=False):
    
    for i in range(len(columns_of_new_data)):
        column = columns_of_new_data[i]
        column_to_update = columns_to_update[i]
        # no need to update id
        if column == 'state/id':
            continue
        new_cell_data = new_data[column][index_in_new_data]        
        update_cell(df, row_index_in_df, column_to_update, new_cell_data, replace_items=replace_items)
        
def get_dict_for_new_ids_from_new_data(new_data, mask):
    return {key:new_data[key][mask] for key in new_data.keys()}
        
def update_df(df, new_data, map_from, map_to, replace_items=False):
    # update df for ids that are already present
    columns_of_new_data = list(new_data.keys())
    columns_to_update = [item.replace(map_from, map_to) for item in columns_of_new_data]
    new_ids = new_data['state/id']
    indices_of_ids_that_are_not_in_df = []
    for index_in_new_data in range(new_ids.shape[0]):
        current_id = new_ids[index_in_new_data]
        row_index_in_df = find_index_of_id_in_df(df, current_id)
        # object id already in df 
        if row_index_in_df >= 0:
            update_row(df, row_index_in_df, columns_to_update, columns_of_new_data, new_data, index_in_new_data, replace_items=replace_items)
        else:
            indices_of_ids_that_are_not_in_df.append(index_in_new_data)
    
    # create a df using for ids that are not already in df
    mask = np.array([False] * new_data['state/id'].shape[0])
    for index in indices_of_ids_that_are_not_in_df:
        mask[index] = True
    dict_for_new_ids_from_new_data = get_dict_for_new_ids_from_new_data(new_data, mask)
    all_column_names_for_df = df.columns
    new_df = pd.DataFrame({'state/id':dict_for_new_ids_from_new_data['state/id']}) 
    for name in all_column_names_for_df:
        new_df[name] = np.nan
    for name in dict_for_new_ids_from_new_data:
        new_df[name.replace(map_from, map_to)] = dict_for_new_ids_from_new_data[name].tolist()    
    
    resulting_df = pd.concat([df, new_df])
    resulting_df['state/id'] = resulting_df['state/id'].astype('int64')
    
    # Transform lists (with is expected to contain only one element) to float
    for desired_key in desired_keys_current:
        resulting_df[desired_key] = resulting_df[desired_key].apply(list_to_float)
    
    return resulting_df

def find_indices_in_whole(df, decoded_whole):
    df_ids = df['state/id']
    indices_in_whole = []
    for df_id in df_ids:
        indices_in_whole.append(np.where(decoded_whole['state/id'] == df_id)[0][0])
    return indices_in_whole

def find_heretofore_gt_trajectories(df, decoded_whole):
    indices_in_whole = find_indices_in_whole(df, decoded_whole)
    heretofore_trajectories = {}

    gt_trajectory_keys = {'state/x', 'state/y'}

    for key in gt_trajectory_keys:
        heretofore_trajectories[key] = decoded_whole[key][indices_in_whole,:CURRENT_INDEX_IN_WHOLE]
    return heretofore_trajectories

def weight_prediction_by_gt(predicted_trajectories, gt_trajectories, gt_weight=0.5):
    result_dict = copy.deepcopy(predicted_trajectories)
    
    assert gt_weight >= 0
    assert gt_weight <= 1
    prediction_weight = 1 - gt_weight
    # preprocess prediction: setting corresponding values in predictions to -1 if gt is invalid
    if gt_weight > 0:
        predicted_trajectories['state/past/x'][gt_trajectories['state/x'] == -1] = -1
        predicted_trajectories['state/past/y'][gt_trajectories['state/y'] == -1] = -1
        
    
    result_dict['state/past/x'] = predicted_trajectories['state/past/x'] * prediction_weight + gt_trajectories['state/x'] * gt_weight
    result_dict['state/past/y'] = predicted_trajectories['state/past/y'] * prediction_weight + gt_trajectories['state/y'] * gt_weight
    
    return result_dict

def get_sdc_future_trajectory(current_index, decoded_whole,sdc_id,num_future_samples=80):
    sdc_id_index_in_whole = np.where(decoded_whole['state/id'] == sdc_id)[0][0]
    
    sdc_future_trajectory_dict = {}
    
    for key in desired_keys_future:
        key_for_whole = key.replace('future/', '')
        sdc_whole_trajectory = decoded_whole[key_for_whole][sdc_id_index_in_whole]
        # no more future
        if current_index+1 >= len(sdc_whole_trajectory):
            sdc_future_trajectory = np.array([-1] * num_future_samples)
        sdc_future_trajectory = sdc_whole_trajectory[current_index+1:current_index+1+num_future_samples]
        # pad if no enough future
        if len(sdc_future_trajectory) < num_future_samples:
            paddings = np.array([-1] * (num_future_samples - len(sdc_future_trajectory)))
            np.hstack([sdc_future_trajectory, paddings])
        sdc_future_trajectory_dict[key] = sdc_future_trajectory
        
    return sdc_future_trajectory_dict

def preserve_recent_past_samples(x, num_past_samples_to_preserve):
    if isNaN(x):
        return [-1] * num_past_samples_to_preserve
    elif len(x) > num_past_samples_to_preserve:
        return x[-num_past_samples_to_preserve:]
    elif len(x) == num_past_samples_to_preserve:
        return x
    else:
        nums_to_pad = num_past_samples_to_preserve - len(x)
        return [-1] * nums_to_pad + x

def float_to_list(x):
    if isNaN(x):
        return [-1]
    else: 
        return [x]
    
def df_column_to_2d_array(df_column):
    df_column = df_column.tolist()
    df_column = [np.array(v) for v in df_column]
    try:
        for i in range(len(df_column)):
            if len(df_column[i].shape) == 2 and df_column[i].shape[1] ==  1 and df_column[i].shape[0] == 1:
                df_column[i] = df_column[i][0]
            
        two_d_array = np.stack(df_column)
    except:
        print([len(k) for k in df_column])
        print(df_column)
        two_d_array = np.stack(df_column)
    if len(two_d_array.shape) > 2:
        two_d_array = two_d_array[:,:,0]
    return two_d_array 

def m2i_inputs_post_processing(res,decoded_example_group):
    ids = res['state/id']
    indices_in_decoded_example = []
    for current_id in ids:
        current_index_in_decoded_example = np.where(decoded_example_group['state/id'] == current_id)[0][0]
        indices_in_decoded_example.append(current_index_in_decoded_example)
    
    
    res['state/past/valid'] = np.float64(res['state/past/x'] != -1)
    res['state/current/valid'] = np.float64(res['state/current/x'] != -1)
    res['state/future/valid'] = np.float64(res['state/future/x'] != -1)
    
    res['state/past/z'] = res['state/past/x'] * 0
    res['state/current/z'] = res['state/current/x'] * 0
    res['state/future/z'] = res['state/future/x'] * 0
    res['state/past/height'] = res['state/past/x'] * 0
    res['state/current/height'] = res['state/current/x'] * 0
    res['state/future/height'] = res['state/future/x'] * 0
    res['state/past/vel_yaw'] = res['state/past/x'] * 0
    res['state/current/vel_yaw'] = res['state/current/x'] * 0
    res['state/future/vel_yaw'] = res['state/future/x'] * 0
    res['state/past/timestamp_micros'] = res['state/past/x'] * 0
    res['state/current/timestamp_micros'] = res['state/current/x'] * 0
    res['state/future/timestamp_micros'] = res['state/future/x'] * 0
    res['state/type'] = np.ones(res['state/id'].shape)
    res['state/objects_of_interest'] = np.ones(res['state/id'].shape, dtype=np.int64)
    res['state/tracks_to_predict'] = np.zeros(res['state/id'].shape)

    for key in res:
        try:
            if res[key].dtype == np.float64:
                res[key] = np.float32(res[key])
        except:
            pass
    return res

def get_inputs_for_M2I(df, sdc_id, num_past_samples_to_preserve=10, num_future_samples=80):
    
    """
    Extract road information and part of trajectories from the 
    complete history to form inputs for M2I. 
    
    Parameters
    ----------
    df: a pandas dataframe 
        It stores the trajectory history, including the first 1.1s of 
        ground truth and trajectories predicted by the M2I module. 
        
    sdc_id: int
        The id of the self-driving car.
        
    num_past_samples_to_preserve: int
        The number of past samples to include in the M2I input.
        
    num_future_samples: int
        The number of future samples to include in the M2I input. 
        
        
    Returns
    -------
    copy_of_decoded_example_group: dict
        A dict that contains the past, current, and future trajectory,
        as well as the road information, which can be used by the M2I
        for future trajectory prediction. Items that are not available 
        in the trajectory history will be filled with invalid values 
        0's and -1's. 
    
    """
    
    
    
    # need id, past, and current
    all_desired_keys = ['state/id'] + desired_keys_past + desired_keys_current
    desired_df = df[all_desired_keys].copy()
    
    # make sure the length of past is 10
    for column in desired_keys_past:
        desired_df[column] = desired_df[column].apply(lambda x: preserve_recent_past_samples(x, num_past_samples_to_preserve=num_past_samples_to_preserve))
        
    
    past_states = {'state/id': np.array(desired_df['state/id'].tolist())}
    current_states = {'state/id': np.array(desired_df['state/id'].tolist())}
    combined_states = {'state/id': np.array(desired_df['state/id'].tolist())}
        
    # combine past with current
    for naive_key in naive_keys:
        past_key = prefix_past + '/' + naive_key
        current_key = prefix_current + '/' + naive_key
        past_item = desired_df[past_key]
        current_item = desired_df[current_key].apply(float_to_list)
        
        # create array of shape [batch_size, num_past_samples_to_preserve]
        past_item = df_column_to_2d_array(past_item)
        
        # create array of shape [batch_size, 1]
        current_item = df_column_to_2d_array(current_item)
        
        past_states[past_key] = past_item
        current_states[current_key] = current_item    
        
        combined_states[past_key] = past_item
        combined_states[current_key] = current_item
        
    # generate a copy of decoded_example_group
    copy_of_decoded_example_group = copy.deepcopy(decoded_example_group)
    # updated states in the copy
    for key in combined_states.keys():
        copy_of_decoded_example_group[key] = combined_states[key]
    # TODO: find index of sdc 
    temp = copy_of_decoded_example_group['state/id'] == sdc_id
    index_of_sdc = np.where(temp)[0][0]
    # update is_sdc
    new_sdc = np.zeros(copy_of_decoded_example_group['state/id'].shape[0])
    new_sdc[index_of_sdc] = 1
    copy_of_decoded_example_group['state/is_sdc'] = new_sdc
    # TODO: decide which locations in is_sdc are invalid and set them to -1's
    # reshape future and fill with dummy data
    for future_key in desired_keys_future:
        past_key = future_key.replace('future', 'past')
        past_shape = combined_states[past_key].shape
        copy_of_decoded_example_group[future_key] = np.zeros([past_shape[0], num_future_samples]) - 1
        
    
    # update the future trajectory for sdc
    future_trajectory_for_sdc = get_sdc_future_trajectory(CURRENT_INDEX_IN_WHOLE,decoded_whole, sdc_id, num_future_samples=num_future_samples)
    for key in future_trajectory_for_sdc:
        new_value = np.zeros(copy_of_decoded_example_group[key].shape) - 1
        new_value[index_of_sdc] = future_trajectory_for_sdc[key]
        copy_of_decoded_example_group[key] = new_value
    # fill missing inputs wiht invalid values
    copy_of_decoded_example_group = m2i_inputs_post_processing(copy_of_decoded_example_group,decoded_example_group)
    
    
    
    # # calculate vel and yaw (past)
    # copy_of_decoded_example_group = velocity_yaw_past(copy_of_decoded_example_group)
    # # calculate vel and yaw (current)
    # copy_of_decoded_example_group = velocity_yaw_current(copy_of_decoded_example_group)
    # # calculate vel and yaw (future)
    # copy_of_decoded_example_group = velocity_yaw_future(copy_of_decoded_example_group)
    
    
    return copy_of_decoded_example_group

def is_the_longest(x, max_length):
    if len(x) == max_length:
        return True
    elif len(x) < max_length:
        return False
    else:
        raise ValueError('Incorrect max length. ' + 'Max length is ' + str(len(x)) + ' instead of ' + str(max_length))

def get_inputs_for_mpc(inputs_for_m2i, predicted_traj_x, predicted_traj_y):
    
    """
    Prepare inputs for MPC.
    
    
    Parameters
    ----------
    inputs_for_m2i: dict
        Inputs for M2I.
        
    
     predicted_traj_x: numpy.ndarray
         x coordinates of future trajectories predicted by M2I.
         
     predicted_traj_y: numpy.ndarray
         y coordinates of future trajectories predicted by M2I.
         
    
    Returns
    -------
    sample_mpc_inputs: dict
        A dict that contains the past, current, and future trajectory,
        as well as the road information, which can be used by the MPC.
        Future trajectories are the same as those predicted by M2I.
    
    """
    
    
    
    sample_mpc_inputs = copy.deepcopy(inputs_for_m2i)
    
    # update future trajectory using m2i outputs
    sample_mpc_inputs['state/future/x'] = predicted_traj_x
    sample_mpc_inputs['state/future/y'] = predicted_traj_y
    
    return sample_mpc_inputs

def filter_inputs_for_mpc(sample_mpc_inputs, th=2.5, max_difference=90):
    
    """
    Filter out trajectories that cannot be processed by MPC, including U-turns
    and trajectories with noisy samples. Noise suppression will be performed 
    first. Trajectories that are noisy and trajectories with U-turns after 
    noise suppression will be filtered out. Additionally, trajectories will
    be ranked by trajectory distances in descending order. Their rankings 
    will be stored in dummy/roadgraph_samples/distance_descending_order. 
    
    Parameters
    ----------
    sample_mpc_inputs: dict
        A dict that contains the past, current, and future trajectory,
        as well as the road information, which can be used by the MPC.
        
    th: float
        The strength of noise suppression. The larger the value is, the
        stronger the noise suppression is. 
        
    max_difference: float
        Maximum allowed direction differences (in degree) for any pair of object
        locations (by x and y) in each future trajectory. If any pair of object 
        location's direction difference is greater than this value, the trajectory
        will be filtered out from MPC inputs. 
        
    
    Returns
    -------
    filtered_mpc_inputs: dict
        A dict that contains the past, current, and future trajectory,
        as well as the road information, which can be used by the MPC.
        Trajectories with U-turns or noise will are removed. 
    
    mpc_valid_indices: numpy.ndarray
        An array of Trues and Falses indicating whether each trajectory 
        in the input sample_mpc_inputs has been filtered out by this function. 
    
    """
    
    # find out valid indices for mpc
    u_turn_indices = find_u_turns(sample_mpc_inputs, th=th, max_difference=max_difference)
    mpc_valid_indices = np.invert(u_turn_indices)
    
    # filter out trajectories whose x's are all -1's 
    for i in range(sample_mpc_inputs['state/future/x'].shape[0]):
        if (sample_mpc_inputs['state/future/x'][i] == -1).all():
            mpc_valid_indices[i] = False
    
    # filter valid indices for mpc
    filtered_mpc_inputs = filter_dict_using_indices(sample_mpc_inputs, mpc_valid_indices)
    
    # rank distance for mpc
    order_descending = rank_distance(filtered_mpc_inputs)
    filtered_mpc_inputs['dummy/roadgraph_samples/distance_descending_order'] = order_descending
    
    return filtered_mpc_inputs, mpc_valid_indices

def process_mpc_outputs(sample_mpc_inputs, mpc_outputs, mpc_valid_indices):
    """
    Remove key-value pairs whose key contain 'dummy.' Additionally, using MPC outputs to 
    update corresponding values in un-filtered MPC inputs. Specifically, some 
    trajectories were filtered out from un-filtered before giving to MPC. For those 
    that were not filtered out, MPC produces updated values. We use these updated 
    values to update corresponding values in un-filtered MPC inputs. 

    Parameters
    ----------
    sample_mpc_inputs: dict
        A dict that contains the past, current, and future trajectory,
        as well as the road information, which can be used by the MPC.
        Future trajectories are the same as those predicted by M2I. It
        contains all trajectories. No trajectory was filtered out. 

    mpc_outputs: dict
        MPC outputs using filtered inputs.

    mpc_valid_indices:
        An array of Trues and Falses indicating whether each trajectory 
        in the input sample_mpc_inputs has was filtered out from filtered 
        MPC inputs. 


    Returns
    -------
    dict_to_updated_df: dict
        A dict that contains the past, current, and future trajectory,
        as well as the road information. Future trajectories are either
        the output of M2I (for filtered out trajectories that could not 
        be processed by MPC) or the output of MPC. This dict can be used
        to update trajectory history. 
        
    """
    mpc_outputs = delete_dummy_from_dict(mpc_outputs)
    dict_to_updated_df = update_dict_using_indices(sample_mpc_inputs, mpc_outputs, mpc_valid_indices)
    return dict_to_updated_df

def get_inputs_for_common_road(df,decoded_example_group, drop_shorter=True, remove_invalid=False):
    # need past
    all_desired_keys = ['state/id'] + desired_keys_past
    desired_df = df[all_desired_keys].copy()
    
    all_past_states = {'state/id': np.array(desired_df['state/id'].tolist())}
    
    # Pad shorter ones
    if not drop_shorter:
        for column in desired_keys_past:
            # pad past
            max_length = np.array([len(x) for x in desired_df[column]]).max()
            desired_df[column] = desired_df[column].apply(lambda x: preserve_recent_past_samples(x, num_past_samples_to_preserve=max_length))
            # get column
            past_item = desired_df[column]
            # create an array of shape []
            past_item = df_column_to_2d_array(past_item)
            # add to dict
            all_past_states[column] = past_item
    # Drop shorter ones
    else:
        max_length = np.array([len(x) for x in desired_df['state/past/x']]).max()
        desired_df = desired_df[desired_df['state/past/x'].apply(lambda x: is_the_longest(x, max_length=max_length))]
        for column in desired_keys_past:
            # get column
            past_item = desired_df[column]
            # create an array of shape []
            past_item = df_column_to_2d_array(past_item)
            # add to dict
            all_past_states[column] = past_item
            
    # Add additional items to dict
    # TODO
    all_past_states['scenario/id'] = decoded_example_group['scenario/id']
    all_past_states['state/is_sdc'] = all_past_states['state/id'] * 0
    all_past_states['state/type'] = decoded_example_group['state/type']
    
    
    # TODO: remove objects whose past or current contain invalid values (-1's)
    if remove_invalid:
        # remove invalid past
        past_mask = all_past_states['state/past/x'][:,0] != -1
        for key in all_past_states.keys():
            if type(all_past_states[key]) != str:
                all_past_states[key] = all_past_states[key][past_mask]
        # remove invalid current
        pass
    
    return all_past_states

def make_dirs(path):
    if  not osp.exists(path):
        os.makedirs(path)

def save_for_commondroad(data,sdc_id,decoded_example_group,save_path,smoothed_save_path):
    # inputs for common road
    sample_common_road_inputs = get_inputs_for_common_road(data,decoded_example_group, drop_shorter=True)
    # re-calculate past
    sample_common_road_inputs = velocity_yaw_past(sample_common_road_inputs)
    for k in sample_common_road_inputs:
        if type(sample_common_road_inputs[k] == np.ndarray):
            sample_common_road_inputs[k] = np.nan_to_num(sample_common_road_inputs[k])
    
    
    # set sdc
    sdc_index_in_common_road_inputs = np.where(sample_common_road_inputs['state/id'] == sdc_id)[0][0]
    sample_common_road_inputs['state/is_sdc'][sdc_index_in_common_road_inputs] = 1
    smoothed_common_road_inputs = noise_suppression(sample_common_road_inputs, 'past', th=MANUAL_TH)
    # write to disk
    with open(save_path, 'wb') as f:
        pickle.dump(sample_common_road_inputs, f)
    with open(smoothed_save_path, 'wb') as f:
        pickle.dump(smoothed_common_road_inputs, f)

def update_df_using_MPC_outputs(df, mpc_outputs, num_future_samples_to_use_as_future):
    
    """
    Update trajectory by adding newly predicted future trajectories to history. 
    Predicted trajectory and current locations of objects will be concatenated 
    to trajectories in trajectory history. The last sample in each predicted 
    trajectory will not be concatenated to trajectories in trajectory history. 
    Instead, they will be used to update the current location of each object. 
    
    
    Parameters
    ----------
    df: pandas.core.frame.DataFrame
    
        Trajectory history which contains object property, velocity, location, 
        and direction data. 
        
    
    mpc_outputs: dict
        A dict that contains the past, current, and future trajectory,
        as well as road information. Future trajectories are either
        the output of M2I (for filtered out trajectories that could not 
        be processed by MPC) or the output of MPC. 
    
    
    num_future_samples_to_use_as_future: int
        The number of samples in future trajectories to discard. Discarded 
        samples will not be used to update trajectory history. 
    
    
    Returns
    -------
    updated_df: pandas.core.frame.DataFrame
        Trajectory history which is updated using newly predicted future trajectories. 
    
    """
    
    
        
    
    
    
    # need id, current, and future
    #all_desired_keys = ['state/id'] + desired_keys_current + desired_keys_future
    
    new_past_states = {'state/id': mpc_outputs['state/id']}
    new_future_states = {'state/id': mpc_outputs['state/id']}
    new_current_states = {'state/id': mpc_outputs['state/id']}
    
    # get current and future:
    for naive_key in naive_keys:
        current_key = prefix_current + '/' + naive_key
        future_key = prefix_future + '/' + naive_key
        new_past_key = 'state/new_past' + '/' + naive_key
        new_future_key = 'state/new_future' + '/' + naive_key
        new_current_key = 'state/new_current' + '/' + naive_key
        
        if current_key not in mpc_outputs.keys() or future_key not in mpc_outputs.keys():
            print('Missing ', current_key, future_key)
            continue
        
        current_item = mpc_outputs[current_key]
        future_item = mpc_outputs[future_key]
        
        new_past_item = np.hstack([current_item, future_item[:, :-num_future_samples_to_use_as_future-1]])
        new_future_item = future_item[:, -num_future_samples_to_use_as_future:]
        new_current_item = future_item[:, -num_future_samples_to_use_as_future-1:-num_future_samples_to_use_as_future]
        
        new_past_states[new_past_key] = new_past_item
        new_future_states[new_future_key] = new_future_item
        new_current_states[new_current_key] = new_current_item
        
    
    # update df using generated dicts
    # 1. concat past and new past
    updated_df = update_df(df, new_past_states, map_from='new_past', map_to='past', replace_items=False)
    # 2. overwrite future with new future
    updated_df = update_df(updated_df, new_future_states, map_from='new_future', map_to='future', replace_items=True)
    # 3. overwrite current with new current
    updated_df = update_df(updated_df, new_current_states, map_from='new_current', map_to='current', replace_items=True)
    # 4. update CURRENT_INDEX_IN_WHOL
    time_passed = new_past_states['state/new_past/x'].shape[1]
    global CURRENT_INDEX_IN_WHOLE
    CURRENT_INDEX_IN_WHOLE += time_passed
    
    return updated_df

def get_ground_truth_future_trajectory(future_length=80):
    indices_in_whole = find_indices_in_whole(df, decoded_whole)
    gt_future_x = decoded_whole['state/x'][indices_in_whole][:,CURRENT_INDEX_IN_WHOLE+1:CURRENT_INDEX_IN_WHOLE+1+future_length]
    gt_future_y = decoded_whole['state/y'][indices_in_whole][:,CURRENT_INDEX_IN_WHOLE+1:CURRENT_INDEX_IN_WHOLE+1+future_length]
    
    return gt_future_x, gt_future_y

def get_ground_truth_future_yaw(future_length=80):
    gt_future_bbox_yaw = decoded_whole['state/bbox_yaw'][indices_in_whole][:,CURRENT_INDEX_IN_WHOLE+1:CURRENT_INDEX_IN_WHOLE+1+future_length]
    gt_future_vel_yaw = decoded_whole['state/vel_yaw'][indices_in_whole][:,CURRENT_INDEX_IN_WHOLE+1:CURRENT_INDEX_IN_WHOLE+1+future_length]
    
    return gt_future_bbox_yaw, gt_future_vel_yaw

def concat_m2i_inference_out(vv_out,vc_out,vp_out):

     
    if vc_out is not None :
        vv_out = np.concatenate([vv_out,vc_out],axis=0)
        
    if vp_out is not None :
        vv_out = np.concatenate([vv_out,vp_out],axis=0)

    return vv_out

def m2i_prediction(res,args,model,MANUAL_TH, trajectory_type=None):
    predicted_traj_x, predicted_traj_y  = None ,None
    if res['state/id'] is  not None:
        # 1. update vel and yaw before running m2i
        res = velocity_yaw_future(velocity_yaw_past(res))
        # 2. remove oscilations in past and current (x and y based on vel x and vel y)
        res = noise_suppression(res, 'past', th=MANUAL_TH)
        # 3. run m2i 
        predicted_traj_x, predicted_traj_y = run_m2i_inference(args, model, res, trajectory_type) 

    return predicted_traj_x, predicted_traj_y 

def add_surplus_value(surplus_keys,src,target):

    for key in surplus_keys:

        if isinstance(src[key],str):
            target[key] = src[key]
        else:
            target[key] = src[key].copy()
            
    return target

def combine_list(src,key,src_shape):
    tmp = None
    for idx, x in enumerate(src):
        if idx == 0:
            tmp = np.array(x[key]).reshape([1,-1])
        else :
            tmp = np.concatenate([tmp,np.array(x[key]).reshape(1,-1)],axis=0)    
    # if tmp is not None and tmp.shape[1]==1:
    #     tmp = tmp.flatten()
    if tmp is not None:
        tmp = tmp.reshape(src_shape)

    return tmp

def combine_as_m2i_input(all_object_keys,parse_data,src):
    tmp={}
    for k in all_object_keys:
        
        tmp[k]=combine_list(parse_data,k,src[k].shape)

    return tmp

def other_keys(res):
    all_object_keys = []
    for k,v in res.items():
        if not k.startswith('state'):
            all_object_keys.append(k)
    return all_object_keys

def object_keys(res):
    all_object_keys = []
    for k,v in res.items():
        if k.startswith('state'):
            all_object_keys.append(k)
    return all_object_keys

def get_state_by_id(_id,src):
    res = {}
    for k,v in src.items():
        if  k.startswith('state'):
            res[k]=v[_id,...]
    return res

def split_data(res):
    res_vehicle= []
    res_cyclist= []
    res_pedestrian= []
    for idx , t in enumerate(res['state/type']):
        if int(t) == 1 :
            res_vehicle.append(get_state_by_id(int(idx),res))
        elif int(t) == 2 :
            res_pedestrian.append(get_state_by_id(int(idx),res))
        elif int(t) == 3 :
            res_cyclist.append(get_state_by_id(int(idx),res))


    all_object_keys = object_keys(res)
    surplus_keys = other_keys(res)
    
    res_vehicle = combine_as_m2i_input(all_object_keys,res_vehicle,res)
    res_cyclist = combine_as_m2i_input(all_object_keys,res_cyclist,res)
    res_pedestrian = combine_as_m2i_input(all_object_keys,res_pedestrian,res)

    res_vehicle=add_surplus_value(surplus_keys,res,res_vehicle)
    res_cyclist=add_surplus_value(surplus_keys,res,res_cyclist)
    res_pedestrian=add_surplus_value(surplus_keys,res,res_pedestrian)
    return res_vehicle,res_cyclist,res_pedestrian

def shorten_future_in_dict(original_dict, desired_future_length):
    
    """
    Reduce the number of samples in future trajectories for MPC.
    
    Parameters
    ----------
    original_dict: dict
        A dict that contains the past, current, and future trajectory,
        as well as the road information, which can be used by the MPC.
        Future trajectories are the same as those predicted by M2I.
      
    desired_future_length: int
        The desired number of samples in each future trajectory. If
        its value is smaller than the number of samples in the 
        future trajectories in original_dict, extra samples at the 
        end of each future trajectory in original_dict will be removed.
        
    Returns
    -------
    shortened_dict: dict
        A dict that contains the past, current, and future trajectory,
        as well as the road information, which can be used by the MPC.
        Future trajectories are shortened. 
    
    """
    
    
    shortened_dict = copy.deepcopy(original_dict)
    
    for key in shortened_dict.keys():
        if 'future' in key and not 'traffic_light' in key:
            shortened_dict[key] = shortened_dict[key][:,:desired_future_length]
    
    return shortened_dict

def use_gt_past_and_current_velocity_yaw_for_the_first_round(res):
    index_mapping = find_indices_in_whole(df, decoded_whole)
    res['state/past/vel_yaw'] = np.float32(decoded_example_group['state/past/vel_yaw'][index_mapping])
    res['state/past/bbox_yaw'] = np.float32(decoded_example_group['state/past/bbox_yaw'][index_mapping])
    res['state/past/velocity_x'] = np.float32(decoded_example_group['state/past/velocity_x'][index_mapping])
    res['state/past/velocity_y'] = np.float32(decoded_example_group['state/past/velocity_y'][index_mapping])
    
    res['state/current/vel_yaw'] = np.float32(decoded_example_group['state/current/vel_yaw'][index_mapping])
    res['state/current/bbox_yaw'] = np.float32(decoded_example_group['state/current/bbox_yaw'][index_mapping])
    res['state/current/velocity_x'] = np.float32(decoded_example_group['state/current/velocity_x'][index_mapping])
    res['state/current/velocity_y'] = np.float32(decoded_example_group['state/current/velocity_y'][index_mapping])
    return res

def add_gt_future_trajectory_as_dummy(res):
    gt_traj_x, gt_traj_y = get_ground_truth_future_trajectory(future_length=80)
    res['dummy/future/gt/x'] = gt_traj_x
    res['dummy/future/gt/y'] = gt_traj_y
    return res, gt_traj_x, gt_traj_y