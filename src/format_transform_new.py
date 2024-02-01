from sre_constants import JUMP
import pandas as pd
import numpy as np
from math import sqrt
import os
import json
import sys
import copy
import matplotlib.pyplot as plt

from IPython import embed
import pickle

def dataset(lanelet_data_path, baidu_data_path):
    lanelet_data_path = '/DATA1/liyang/M2I_2/baidu/yizhuang#1/1650251580.01-1650251760.00/yizhuang_hdmap1_transform1.json'
    baidu_data_path = '/DATA1/liyang/M2I_2/baidu/yizhuang#1/1650251580.01-1650251760.00'
    frame_num_flag = sys.maxsize  # set show frame numbers
    time_interval = 0.10  # set time interval
    ori_time_interval = 0.05  # The timestamp on Baidu's dataset is about 0.5 seconds
    jump_step = int(np.ceil(time_interval / ori_time_interval))

    obstacles_data_path = os.path.join(baidu_data_path, 'obstacles.csv')
    # traffic_light_data_path = os.path.join(baidu_data_path, 'traffic_lights.csv')
    # rsu_map_path = os.path.join('/'.join(baidu_data_path.split("/")[:3]), 'rsu_map/yz' + baidu_data_path.split("/")[-2][-1], 'rsu_map.xml')

    # bzip2 -d obstacles.csv.bz2 
    output_path = os.path.join('data/commonroad_dataset', '/'.join(baidu_data_path.split('/')[2:-1]))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # get map center coords
    obstacles = pd.read_csv(obstacles_data_path, sep=',')
    # traffic_ligths = pd.read_csv(traffic_light_data_path, sep=',')

    map_center_x = obstacles['area_map.feature_position.x'].iloc[1]
    map_center_y = obstacles['area_map.feature_position.y'].iloc[1]
    map_center_z = obstacles['area_map.feature_position.z'].iloc[1]

    x_start = obstacles['perception_obstacle.position.x'].min()
    x_end = obstacles['perception_obstacle.position.x'].max()
    y_start = obstacles['perception_obstacle.position.y'].min()
    y_end = obstacles['perception_obstacle.position.y'].max()

    obstacles_len = (obstacles['header.sequence_num'].max() - obstacles['header.sequence_num'].min()) // jump_step
    file_name = baidu_data_path.split('/')[-2] + '_' + baidu_data_path.split('/')[-1] + '_MaxLen_' + str(obstacles_len) + '.xml'

    # empty file
    with open(os.path.join(output_path, file_name), 'w') as output_file:
        output_file.write('')
    with open(lanelet_data_path, 'r') as f:
        lanelet_baidu_data = json.load(f)


    sequence_num_start = obstacles['header.sequence_num'].min()

    dynamicObstacle = obstacles[(obstacles['perception_obstacle.sub_type'] != 'UNKNOWN_UNMOVABLE') & (obstacles['perception_obstacle.sub_type'] != 'TRAFFICCONE')]
    staticObstacle = obstacles[(obstacles['perception_obstacle.sub_type'] == 'UNKNOWN_UNMOVABLE') | (obstacles['perception_obstacle.sub_type'] == 'TRAFFICCONE')]
    dynamicObstacle_baidu_list = ['CAR', 'VAN', 'CYCLIST', 'MOTORCYCLIST', 'PEDESTRIAN', 'TRICYCLIST', 'BUS', 'TRUCK']
    dynamicObstacle_croad_list = ['car', 'truck', 'bicycle', 'motorcycle', 'pedestrian', 'truck', 'bus', 'truck']

    dynamicObstacle_id = list(dynamicObstacle['perception_obstacle.id'].drop_duplicates())
    staticObstacle_id = list(staticObstacle['perception_obstacle.id'].drop_duplicates())


    def process_data():
    # 处理了被错误识别为static obstacle 的物体
        for d_id in staticObstacle_id:
            if d_id in dynamicObstacle_id:
                # print("Duplicate ID: {}...".format(d_id))
                sub_type = list(dynamicObstacle[dynamicObstacle['perception_obstacle.id'] == d_id]['perception_obstacle.sub_type'])[0]
                staticObstacle.loc[staticObstacle['perception_obstacle.id'] == d_id, 'perception_obstacle.sub_type'] = sub_type
                s_obstacle_data = staticObstacle[staticObstacle['perception_obstacle.id'] == d_id]
                # 移除staticObstacle_id中的id并删除staticObstacle中错误id所在的所有行
                staticObstacle_id.remove(d_id)
                staticObstacle = staticObstacle.drop(index=staticObstacle[(staticObstacle['perception_obstacle.id'] == d_id)].index.tolist())
                staticObstacle = staticObstacle.reset_index(drop=True)
                dynamicObstacle = dynamicObstacle.append(s_obstacle_data, ignore_index=True)


    def set_lanelet():
        LaneType = {
            "NONE": 0,
            "HIGHWAY_DRIVING":1,
            "CITY_DRIVING" : 2,
            "SHOULDER" : 1,
            "BIKING" : 3,
            "SIDEWALK" : 4,
            "BORDER" : 15,
            "RESTRICTED" : 17,
            "PARKING" : 17,
            "BIDIRECTIONAL" : 2,
            "MEDIAN" : 16,
            "ROADWORKS" : 17,
            "TRAM" : 4,
            "TAIL" : 4,
            "ENTRY" : 4,
            "EXIT" : 4,
            "OFFRAMP" : 4,
            "ONRAMP" : 4,
            "LEFT_TURN_WAITING_ZONE" : 17,
            "EMERGENCY_LANE" : 2,
            "ROUNDABOUT" : 4,
            "REVERSIBLE" : 4,
            "DIRECTION_VARIABLE" : 2,
            "BUS_LANE" : 2,
            "TOLL_LANE" : 2,
            "NO_TURN_WAITING_ZONE" : 19}
        lane = 0
        _xyz = []
        _id = []
        _type = []
        cnt = []
        for lanelet_id in lanelet_baidu_data:
            centerline = lanelet_baidu_data[lanelet_id]['centerline']
            lane_type = lanelet_baidu_data[lanelet_id]['lane_type']
            # has_traffic_control = lanelet_baidu_data[lanelet_id]['has_traffic_control']
            # left_boundary = lanelet_baidu_data[lanelet_id]['left_boundary']
            # right_boundary = lanelet_baidu_data[lanelet_id]['right_boundary']
            xb = float(centerline[0].split(',')[0].split('(')[1])
            yb = float(centerline[0].split(',')[1].split(')')[0])
            xe = float(centerline[-1].split(',')[0].split('(')[1])
            ye = float(centerline[-1].split(',')[1].split(')')[0])
            if  (xb<x_start-20) or (yb<y_start-20) or (xe>x_end+20) or (ye>y_end+20):
                lane = lane + 1
                continue
            _xyz.extend(centerline)
            _id.extend((np.ones((len(centerline))) * lane).tolist())
            _type.extend((np.ones((len(centerline))) * float(LaneType[lane_type])).tolist())
            cnt.append(len(_xyz))
            lane = lane + 1
        dir = np.zeros((len(_xyz),3), dtype=np.float32)
        xyz = -1 * np.ones((len(_xyz),3), dtype=np.float32)
        for i in range(len(_xyz)):
            xyz[i,0] = float(_xyz[i].split(',')[0].split('(')[1]) - map_center_x
            xyz[i,1] = float(_xyz[i].split(',')[1].split(')')[0]) - map_center_y
            if i > 0:
                dir[i-1,0] = (xyz[i,0] - xyz[i-1,0])/sqrt((xyz[i,0] - xyz[i-1,0])**2+(xyz[i,1] - xyz[i-1,1])**2)
                dir[i-1,1] = (xyz[i,1] - xyz[i-1,1])/sqrt((xyz[i,0] - xyz[i-1,0])**2+(xyz[i,1] - xyz[i-1,1])**2)
                if i in cnt:
                    dir[i-1,0] = 0
                    dir[i-1,1] = 0

        id = (np.array(_id)-np.array(_id).min())[:,None]
        type = np.array(_type).astype(np.float)
        valid = np.ones((len(id), 1))

        x = xyz[:,0]
        y = xyz[:,1]
        plt.figure(figsize=(50, 50))
        plt.title("yizhuang_hdmap14.json")
        plt.scatter(x, y, s=50, c='b')
        plt.savefig('/DATA1/liyang/M2I_2/image/road4.png')
        
        roadgraph = {'roadgraph_samples/dir':dir, 'roadgraph_samples/id':id, 'roadgraph_samples/type':type, 
                    'roadgraph_samples/xyz':xyz, 'roadgraph_samples/valid':valid.astype(np.int64)}
        return roadgraph



    def set_obstacle():
        LEN = obstacles_len + 1 
        obj = 0
        bbox_yaw = -1 * np.ones((512,LEN), dtype=np.float32)
        length = -1 * np.ones((512,LEN), dtype=np.float32)
        height = -1 * np.ones((512,LEN), dtype=np.float32)
        width = -1 * np.ones((512,LEN), dtype=np.float32)
        timestamp_micros = -1 * np.ones((512,LEN))
        valid = np.zeros((512,LEN))
        vel_yaw = -1 * np.ones((512,LEN))
        velocity_x = -1 * np.ones((512,LEN), dtype=np.float32)
        velocity_y = -1 * np.ones((512,LEN), dtype=np.float32)
        x = -1 * np.ones((512,LEN), dtype=np.float32)
        y = -1 * np.ones((512,LEN), dtype=np.float32)
        z = -1 * np.ones((512,LEN), dtype=np.float32)
        id = -1 * np.ones((512))
        is_sdc = -1 * np.ones((512))
        objects_of_interest = 0 * np.ones((512), dtype=np.int64)
        tracks_to_predict = -1 * np.ones((512))
        type = -1 * np.ones((512))
        #

        for dObstacle_id in dynamicObstacle_id:
            # print(dObstacle_id)
            d_obstacle_data = dynamicObstacle[dynamicObstacle['perception_obstacle.id'] == dObstacle_id]
            if len(d_obstacle_data) < 3:  # If the number of outgoing frames is less than 3, remov it directly
                continue

            # 按照时间戳排序
            d_obstacle_data = d_obstacle_data.sort_values('header.sequence_num', ascending=True)
            d_obstacle_data = d_obstacle_data.reset_index(drop=True)

            sub_type = str(d_obstacle_data.iloc[0]['perception_obstacle.sub_type'])
            if sub_type not in dynamicObstacle_baidu_list:
                continue
            
            id[obj] = dObstacle_id
            if sub_type == 'UNKNOW':
                type[obj] = 0
            elif sub_type == 'CAR' or 'VAN' or 'BUS' or 'TRUCK':
                type[obj] = 1
            elif sub_type == 'CYCLIST' or 'MOTORCYCLIST' or 'TRICYCLIST':
                type[obj] = 3
            elif sub_type == 'PEDESTRIAN':
                type[obj] = 2
            is_sdc[obj] = 0
            if obj == 68: # 1535178为主车，是第69个obj
                is_sdc[obj] = 1     
                objects_of_interest[obj] = 1 
                tracks_to_predict[obj] = 1
            if obj == 87: # 1535205为从车，是第88个obj
                is_sdc[obj] = 0     
                objects_of_interest[obj] = 1 
                tracks_to_predict[obj] = 1
            relation_gt = [1535178, ]
            # if obj == 66 or 74 or 76 or 81 or 83: # 较为重要的物体 1535176 1535187 1535196 1535199 1535185
            #     is_sdc[obj] = 0     
            #     objects_of_interest[obj] = 0 
            #     tracks_to_predict[obj] = 1
            
            idx = dynamicObstacle_baidu_list.index(sub_type)
            sub_type = dynamicObstacle_croad_list[idx]
            # 在进行轨迹定义的时候，首先需要明确，此时的obstacle_data是从索引1开始的
            d_obstacle_data_len = len(d_obstacle_data)
            start = d_obstacle_data.iloc[0]['header.sequence_num']
            for i in range(0, d_obstacle_data_len, jump_step):
                j = (start-sequence_num_start)//jump_step + i//jump_step

                x[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.position.x'] - float(map_center_x)
                y[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.position.y'] - float(map_center_y)
                z[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.position.z'] - float(map_center_z)
                timestamp_micros[obj,j] = d_obstacle_data.iloc[i]['header.sequence_num']
                height[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.height']
                width[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.width']
                length[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.length']
                valid[obj,j] = 1
                bbox_yaw[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.theta']
                velocity_x[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.velocity.x']
                velocity_y[obj,j] = d_obstacle_data.iloc[i]['perception_obstacle.velocity.y']
                vel_yaw[obj,j] = np.arctan(velocity_y[obj,j]/velocity_x[obj,j])
            obj = obj + 1
        

        # bbox_yaw = tf.convert_to_tensor(bbox_yaw)
        trajectory = {'state/bbox_yaw':bbox_yaw, 'state/bbox_yaw':bbox_yaw, 'state/x':x, 'state/y':y, 'state/z':z, 
                            'state/height': height, 'state/width':width, 'state/length':length, 'state/vel_yaw':vel_yaw, 
                            'state/valid': valid, 'state/velocity_x':velocity_x, 'state/velocity_y':velocity_y, 
                            'state/timestamp_micros':timestamp_micros, 'state/id':id, 'state/is_sdc':is_sdc, 'state/type':type,
                            'state/objects_of_interest':objects_of_interest, 'state/tracks_to_predict':tracks_to_predict,
                            }
        for k,v in trajectory.items():
            if len(v.shape) == 2:
                trajectory[k][[68,0],:] = v[[0,68],:]
                trajectory[k][[87,1],:] = v[[1,87],:]
            else:
                trajectory[k][[68,0]] = v[[0,68]]
                trajectory[k][[87,1]] = v[[1,87]]

        return trajectory


    def set_traffic_lights():
        LEN = obstacles_len + 1 
        id = -1 * np.ones((LEN,16))
        state = -1 * np.ones((LEN,16))
        valid = -1 * np.ones((LEN,16))
        x = -1 * np.ones((LEN,16), dtype=np.float32)
        y = -1 * np.ones((LEN,16), dtype=np.float32)
        z = -1 * np.ones((LEN,16), dtype=np.float32)

        traffic_light = {'traffic_light_state/id':id, 'traffic_light_state/state':state, 'traffic_light_state/valid':valid,
                        'traffic_light_state/x':x, 'traffic_light_state/y':y, 'traffic_light_state/z':z}
        
        return traffic_light
    
    def split_decoded_data_bk(decoded_example):
        data_total_tick = decoded_example['state/x'].shape[1]

        past = 10
        current = 1
        future = 80
        single_all_tick = past + current + future
        overlap = 50
        delta = single_all_tick - overlap

        split_count = (data_total_tick - single_all_tick) // delta + 1

        decoded_example_group = []

        for i in range(split_count):
            new_decoded_example = {}

            begin_tick = i * delta
            past_end_tick = begin_tick + past
            current_end_tick = past_end_tick + current
            end_tick = begin_tick + single_all_tick
            assert end_tick == current_end_tick + future

            for k,v in decoded_example.items():
                if k == 'scenario/id':
                    new_decoded_example[k] = copy.copy(decoded_example[k])
                elif len(v.shape) == 2:
                    if v.shape[0] == data_total_tick:
                        new_decoded_example[k.split('/')[0]+'/past/'+k.split('/')[1]] = decoded_example[k].copy()[begin_tick:past_end_tick, :]
                        new_decoded_example[k.split('/')[0]+'/current/'+k.split('/')[1]] = decoded_example[k].copy()[past_end_tick:current_end_tick, :]
                        new_decoded_example[k.split('/')[0]+'/future/'+k.split('/')[1]] = decoded_example[k].copy()[current_end_tick:end_tick, :]
                    elif v.shape[1] == data_total_tick:
                        new_decoded_example[k.split('/')[0]+'/past/'+k.split('/')[1]] = decoded_example[k].copy()[:, begin_tick:past_end_tick]
                        new_decoded_example[k.split('/')[0]+'/current/'+k.split('/')[1]] = decoded_example[k].copy()[:, past_end_tick:current_end_tick]
                        new_decoded_example[k.split('/')[0]+'/future/'+k.split('/')[1]] = decoded_example[k].copy()[:, current_end_tick:end_tick]
                    else:
                        new_decoded_example[k] = decoded_example[k].copy()
                else:
                    new_decoded_example[k] = decoded_example[k].copy()

            decoded_example_group.append(new_decoded_example)

        return decoded_example_group

    def split_decoded_data(decoded_example):
        data_total_tick = decoded_example['state/x'].shape[1]

        past = 10
        current = 1
        future = 80
        single_all_tick = past + current + future
        overlap = 50
        delta = single_all_tick - overlap

        split_count = (data_total_tick - single_all_tick) // delta + 1

        decoded_example_group = []

        for i in range(1):
            new_decoded_example = {}

            begin_tick = i * delta + 173  # 173
            past_end_tick = begin_tick + past
            current_end_tick = past_end_tick + current
            end_tick = begin_tick + single_all_tick
            assert end_tick == current_end_tick + future

            for k,v in decoded_example.items():
                if k == 'scenario/id':
                    new_decoded_example[k] = copy.copy(decoded_example[k])
                elif len(v.shape) == 2:
                    if v.shape[0] == data_total_tick:
                        new_decoded_example[k.split('/')[0]+'/past/'+k.split('/')[1]] = decoded_example[k].copy()[begin_tick:past_end_tick, :]
                        new_decoded_example[k.split('/')[0]+'/current/'+k.split('/')[1]] = decoded_example[k].copy()[past_end_tick:current_end_tick, :]
                        new_decoded_example[k.split('/')[0]+'/future/'+k.split('/')[1]] = decoded_example[k].copy()[current_end_tick:end_tick, :]
                    elif v.shape[1] == data_total_tick:
                        new_decoded_example[k.split('/')[0]+'/past/'+k.split('/')[1]] = decoded_example[k].copy()[:, begin_tick:past_end_tick]
                        new_decoded_example[k.split('/')[0]+'/current/'+k.split('/')[1]] = decoded_example[k].copy()[:, past_end_tick:current_end_tick]
                        new_decoded_example[k.split('/')[0]+'/future/'+k.split('/')[1]] = decoded_example[k].copy()[:, current_end_tick:end_tick]
                    else:
                        new_decoded_example[k] = decoded_example[k].copy()
                else:
                    new_decoded_example[k] = decoded_example[k].copy()

            decoded_example_group.append(new_decoded_example)

        # embed()
        return decoded_example_group

    process_data()
    roadgraph = set_lanelet()
    trajectory = set_obstacle()
    traffic_light = set_traffic_lights()

    decoded_example = roadgraph.copy()
    decoded_example.update(trajectory)
    decoded_example = decoded_example.copy()
    decoded_example.update(traffic_light)
    scenario = baidu_data_path[-38:]
    decoded_example['scenario/id'] = scenario
    
    decoded_example_group = split_decoded_data(decoded_example)




    # # temp
    # ego_car_traj = np.concatenate([decoded_example_group[0]['state/future/x'][0].reshape(-1,1), 
    #                 decoded_example_group[0]['state/future/y'][0].reshape(-1,1)], axis=-1)
    # a = {'ego': ego_car_traj}
    # with open('ego.pickle', 'wb') as f:
    #     pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)


    return decoded_example_group

if __name__ == '__main__':
    lanelet_data_path = None
    baidu_data_path = None
    dataset(lanelet_data_path, baidu_data_path)
