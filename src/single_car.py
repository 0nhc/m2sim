import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
import random
import pickle
import cubic_spline_planner
import copy

# 汽车状态
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None

# 汽车MPC控制器
class SINGLE_MPC:
    """
    MPC controller for a single car
    """
    def __init__(self):
        self.NX = 4  # x = x, y, v, yaw
        self.NU = 2  # a = [accel, steer]
        self.T = 5  # horizon length

        # mpc parameters
        self.R = np.diag([0.1, 0.1])  # input cost matrix
        self.Rd = np.diag([0.1, 1.0])  # input difference cost matrix
        self.Q = np.diag([1.0, 1.0, 0.5, 1.0])  # state cost matrix
        self.Qf = self.Q  # state final matrix
        self.GOAL_DIS = 5.0  # goal distance
        self.STOP_SPEED = 0.5 / 3.6  # stop speed
        self.MAX_TIME = 20.0  # max simulation time

        # iterative paramter
        self.MAX_ITER = 2  # Max iteration
        self.DU_TH = 0.1  # iteration finish param

        self.TARGET_SPEED = 40.0 / 3.6  # [m/s] target speed
        self.N_IND_SEARCH = 10  # Search index number

        self.DT = 0.2  # [s] time tick

        # Vehicle parameters
        self.LENGTH = 4.5  # [m]
        self.WIDTH = 2.0  # [m]
        self.BACKTOWHEEL = 1.0  # [m]
        self.WHEEL_LEN = 0.3  # [m]
        self.WHEEL_WIDTH = 0.2  # [m]
        self.TREAD = 0.7  # [m]
        self.WB = 2.5  # [m]

        self.MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
        self.MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
        self.MAX_SPEED = 60.0 / 3.6  # maximum speed [m/s]
        self.MIN_SPEED = 0.0 / 3.6  # minimum speed [m/s]
        self.MAX_ACCEL = 1.0  # maximum accel [m/ss]

        self.XY_GOAL_TOLERANCE = self.GOAL_DIS # [m]
        self.DL = 1.0  # course tick [m]

        self.SHOW_ANIMATION = True
        self.SHOW_POTENTIAL_FIELD = True
        self.OBSTACLE_AVOIDANCE = True


    def pi_2_pi(self, angle):
        while(angle > math.pi):
            angle = angle - 2.0 * math.pi

        while(angle < -math.pi):
            angle = angle + 2.0 * math.pi

        return angle

    def invalid_filter(self, x, y):
        '''
        Description: 去除(-1, -1)的无效坐标，替换成前一个有效坐标值
        '''
        x = x.tolist()
        y = y.tolist()
        for i in range(1, len(x)):
            if(x[-1] == -1 and y[-1] == -1):
                x.pop()
                y.pop()
        # 为了计算曲率，确保waypoints长度为3
        while(len(x)<3):
            x.append(x[-1]+random.random())
            y.append(y[-1]+random.random())
        # if(x[0]==-1 and y[0]==-1):
        #     print('invalid trajectory')
        #     return None, None
        # else:
        #     return x, y
        return x, y

    def get_linear_model_matrix(self, v, phi, delta):

        A = np.zeros((self.NX, self.NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.DT * math.cos(phi)
        A[0, 3] = - self.DT * v * math.sin(phi)
        A[1, 2] = self.DT * math.sin(phi)
        A[1, 3] = self.DT * v * math.cos(phi)
        A[3, 2] = self.DT * math.tan(delta) / self.WB

        B = np.zeros((self.NX, self.NU))
        B[2, 0] = self.DT
        B[3, 1] = self.DT * v / (self.WB * math.cos(delta) ** 2)

        C = np.zeros(self.NX)
        C[0] = self.DT * v * math.sin(phi) * phi
        C[1] = - self.DT * v * math.cos(phi) * phi
        C[3] = - self.DT * v * delta / (self.WB * math.cos(delta) ** 2)

        return A, B, C


    def plot_car(self, x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

        outline = np.array([[-self.BACKTOWHEEL, (self.LENGTH - self.BACKTOWHEEL), (self.LENGTH - self.BACKTOWHEEL), -self.BACKTOWHEEL, -self.BACKTOWHEEL],
                            [self.WIDTH / 2, self.WIDTH / 2, - self.WIDTH / 2, -self.WIDTH / 2, self.WIDTH / 2]])

        fr_wheel = np.array([[self.WHEEL_LEN, -self.WHEEL_LEN, -self.WHEEL_LEN, self.WHEEL_LEN, self.WHEEL_LEN],
                            [-self.WHEEL_WIDTH - self.TREAD, -self.WHEEL_WIDTH - self.TREAD, self.WHEEL_WIDTH - self.TREAD, self.WHEEL_WIDTH - self.TREAD, -self.WHEEL_WIDTH - self.TREAD]])

        rr_wheel = np.copy(fr_wheel)

        fl_wheel = np.copy(fr_wheel)
        fl_wheel[1, :] *= -1
        rl_wheel = np.copy(rr_wheel)
        rl_wheel[1, :] *= -1

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
        Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                        [-math.sin(steer), math.cos(steer)]])

        fr_wheel = (fr_wheel.T.dot(Rot2)).T
        fl_wheel = (fl_wheel.T.dot(Rot2)).T
        fr_wheel[0, :] += self.WB
        fl_wheel[0, :] += self.WB

        fr_wheel = (fr_wheel.T.dot(Rot1)).T
        fl_wheel = (fl_wheel.T.dot(Rot1)).T

        outline = (outline.T.dot(Rot1)).T
        rr_wheel = (rr_wheel.T.dot(Rot1)).T
        rl_wheel = (rl_wheel.T.dot(Rot1)).T

        outline[0, :] += x
        outline[1, :] += y
        fr_wheel[0, :] += x
        fr_wheel[1, :] += y
        rr_wheel[0, :] += x
        rr_wheel[1, :] += y
        fl_wheel[0, :] += x
        fl_wheel[1, :] += y
        rl_wheel[0, :] += x
        rl_wheel[1, :] += y

        plt.plot(np.array(outline[0, :]).flatten(),
                np.array(outline[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fr_wheel[0, :]).flatten(),
                np.array(fr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rr_wheel[0, :]).flatten(),
                np.array(rr_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(fl_wheel[0, :]).flatten(),
                np.array(fl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(np.array(rl_wheel[0, :]).flatten(),
                np.array(rl_wheel[1, :]).flatten(), truckcolor)
        plt.plot(x, y, "*")


    def update_state(self, state, a, delta):

        # input check
        if delta >= self.MAX_STEER:
            delta = self.MAX_STEER
        elif delta <= -self.MAX_STEER:
            delta = -self.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.DT
        state.y = state.y + state.v * math.sin(state.yaw) * self.DT
        state.yaw = state.yaw + state.v / self.WB * math.tan(delta) * self.DT
        state.v = state.v + a * self.DT

        if state.v > self.MAX_SPEED:
            state.v = self.MAX_SPEED
        elif state.v < self.MIN_SPEED:
            state.v = self.MIN_SPEED

        return state


    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()


    def calc_nearest_index(self, state, cx, cy, cyaw, pind):

        dx = [state.x - icx for icx in cx[pind:(pind + self.N_IND_SEARCH)]]
        dy = [state.y - icy for icy in cy[pind:(pind + self.N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)

        ind = d.index(mind) + pind

        mind = math.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind


    def predict_motion(self, x0, oa, od, xref):
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.T + 1)):
            state = self.update_state(state, ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw

        return xbar


    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od, xref2, dref2):
        """
        MPC contorl with updating operational point iteraitvely
        """

        if oa is None or od is None:
            oa = [0.0] * self.T
            od = [0.0] * self.T

        for i in range(self.MAX_ITER):
            xbar = self.predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref, xref2, dref2)
            if oa is not None and od is not None:
                du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
                if du <= self.DU_TH:
                    break
        else:
            pass
            #print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov


    def linear_mpc_control(self, xref, xbar, x0, dref, xref2, dref2):
        """
        linear mpc control

        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """

        x = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))

        cost = 0.0
        constraints = []

        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)
                cost += cvxpy.quad_form(xref2[:, t] - x[:, t], self.Q)

            A, B, C = self.get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                                self.MAX_DSTEER * self.DT]

        cost += cvxpy.quad_form(xref[:, self.T] - x[:, self.T], self.Qf)
        cost += cvxpy.quad_form(xref2[:, self.T] - x[:, self.T], self.Qf)

        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.MAX_SPEED]
        constraints += [x[2, :] >= self.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL:
            ox = self.get_nparray_from_matrix(x.value[0, :])
            oy = self.get_nparray_from_matrix(x.value[1, :])
            ov = self.get_nparray_from_matrix(x.value[2, :])
            oyaw = self.get_nparray_from_matrix(x.value[3, :])
            oa = self.get_nparray_from_matrix(u.value[0, :])
            odelta = self.get_nparray_from_matrix(u.value[1, :])

        elif prob.status == cvxpy.OPTIMAL_INACCURATE:
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov


    def calc_ref_trajectory(self, state, cx, cy, cyaw, ck, sp, dl, pind):
        xref = np.zeros((self.NX, self.T + 1))
        xref2 = np.zeros((self.NX, self.T + 1))
        dref = np.zeros((1, self.T + 1))
        dref2 = np.zeros((1, self.T + 1))
        ncourse = len(cx)

        ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        xref2[0, 0] = cx[ind] * 1.2
        xref2[1, 0] = cy[ind] * 1.2
        xref2[2, 0] = sp[ind] * 1.2
        xref2[3, 0] = cyaw[ind] * 1.2

        dref[0, 0] = 0.0  # steer operational point should be 0
        dref2[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(self.T + 1):
            travel += abs(state.v) * self.DT
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                xref[0, i] = cx[ind + dind]
                xref[1, i] = cy[ind + dind]
                xref[2, i] = sp[ind + dind]
                xref[3, i] = cyaw[ind + dind]
                xref2[0, i] = cx[ind + dind] * 1.2
                xref2[1, i] = cy[ind + dind] * 1.2
                xref2[2, i] = sp[ind + dind] * 1.2
                xref2[3, i] = cyaw[ind + dind] * 1.2

                dref[0, i] = 0.0
                dref2[0, i] = 0.0
            else:
                xref[0, i] = cx[ncourse - 1]
                xref[1, i] = cy[ncourse - 1]
                xref[2, i] = sp[ncourse - 1]
                xref[3, i] = cyaw[ncourse - 1]
                xref2[0, i] = cx[ncourse - 1] * 1.2
                xref2[1, i] = cy[ncourse - 1] * 1.2
                xref2[2, i] = sp[ncourse - 1] * 1.2
                xref2[3, i] = cyaw[ncourse - 1] * 1.2

                dref[0, i] = 0.0
                dref2[0, i] = 0.0

        return xref, ind, dref, xref2, dref2


    def check_goal(self, state, goal, tind, nind):

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        d = math.hypot(dx, dy)

        isgoal = (d <= self.GOAL_DIS)

        if abs(tind - nind) >= 5:
            isgoal = False

        isstop = (abs(state.v) <= self.STOP_SPEED)

        if isgoal and isstop:
            return True

        return False


    def calc_speed_profile(self, cx, cy, cyaw, target_speed):
        '''
        Description: 计算每一个坐标对应的速度方向
        '''
        speed_profile = [target_speed] * len(cx)
        direction = 1.0  # forward

        # Set stop point
        for i in range(len(cx) - 1):
            dx = cx[i + 1] - cx[i]
            dy = cy[i + 1] - cy[i]

            move_direction = math.atan2(dy, dx)

            if dx != 0.0 and dy != 0.0:
                dangle = abs(self.pi_2_pi(move_direction - cyaw[i]))
                if dangle >= math.pi / 4.0:
                    direction = -1.0
                else:
                    direction = 1.0

            if direction != 1.0:
                speed_profile[i] = - target_speed
            else:
                speed_profile[i] = target_speed

        speed_profile[-1] = 0.0

        return speed_profile


    def smooth_yaw(self, yaw):

        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]

            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

        return yaw


    def get_switch_back_course(self, dl, ax, ay):
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=dl)

        return cx, cy, cyaw, ck


    def calc_v(self, distance, v):
        ratio = 1.25
        if(v<=12.0):
            detect_range = 12.0*ratio
        else:
            detect_range = v*ratio
        obstacle_distance_range = detect_range #meter
        if(distance<=obstacle_distance_range):
            return (obstacle_distance_range - distance)
        else:
            return 0


    def calc_yaw_and_k(self, wp_x, wp_y):
        yaw = [0] * len(wp_x)
        k = [0] * len(wp_x)
        for i in range(len(wp_x)):
            if i == 0:
                dx = wp_x[i+1] - wp_x[i]
                dy = wp_y[i+1] - wp_y[i]
                ddx = wp_x[2] + wp_x[0] - 2*wp_x[1]
                ddy = wp_y[2] + wp_y[0] - 2*wp_x[1]
            elif i == (len(wp_x)-1):
                dx = wp_x[i] - wp_x[i-1]
                dy = wp_y[i] - wp_y[i-1]
                ddx = wp_x[i] + wp_x[i-2] - 2*wp_x[i-1]
                ddy = wp_y[i] + wp_y[i-2] - 2*wp_y[i-1]
            else:      
                dx = wp_x[i+1] - wp_x[i]
                dy = wp_y[i+1] - wp_y[i]
                ddx = wp_x[i+1] + wp_x[i-1] - 2*wp_x[i]
                ddy = wp_y[i+1] + wp_y[i-1] - 2*wp_y[i]
            yaw[i]=math.atan2(dy,dx)
            # 计算曲率:设曲线r(t) =(x(t),y(t)),则曲率k=(x'y" - x"y')/((x')^2 + (y')^2)^(3/2).
            # 参考：https://blog.csdn.net/weixin_46627433/article/details/123403726
            k[i]=(ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2)) # 曲率k计算

        return yaw, k


    def setup(self, original_data, index):
        self.data = copy.deepcopy(original_data)
        self.x_data = self.data['state/future/x'][index]
        self.y_data = self.data['state/future/y'][index]

        #self.x_data = self.x_data[6:]
        #self.y_data = self.y_data[6:]

        self.x_data, self.y_data = self.invalid_filter(self.x_data, self.y_data)
        self.length_data = self.data['state/past/length'][index]
        self.width_data = self.data['state/past/width'][index]

        # 计算汽车的长
        self.average_length = 0
        available_length_num = 0
        for length in self.length_data:
            if(length>0):
                self.average_length += length
                available_length_num += 1
        if(available_length_num == 0):
            self.average_length = 3
        else:
            self.average_length = self.average_length/available_length_num

        # 计算汽车的宽
        self.average_width = 0
        available_width_num = 0
        for width in self.width_data:
            if(width>0):
                self.average_width += width
                available_width_num += 1
        if(available_width_num==0):
            self.average_width = 2
        else:
            self.average_width = self.average_width/available_width_num
        
        # try:
        #     # 通过三次样条曲线拟合轨迹，平滑处理
        #     self.cx, self.cy, self.cyaw, self.ck = self.get_switch_back_course(self.DL, self.x_data, self.y_data)
        # except:
        # # 直接拿m2i输入的轨迹来用
        #     self.cx, self.cy = self.x_data, self.y_data
        #     self.cyaw, self.ck = self.calc_yaw_and_k(self.cx, self.cy)
        #     self.cyaw = self.smooth_yaw(self.cyaw)

        # 直接拿m2i输入的轨迹来用
        self.cx, self.cy = self.x_data, self.y_data
        self.cyaw, self.ck = self.calc_yaw_and_k(self.cx, self.cy)
        self.cyaw = self.smooth_yaw(self.cyaw)

        self.sp = self.calc_speed_profile(self.cx, self.cy, self.cyaw, self.TARGET_SPEED)
        # 计算初始速度
        if(len(self.cx) >= 2):
            self.initial_vx = (self.cx[1] - self.cx[0])/self.DT
            self.initial_vy = (self.cy[1] - self.cy[0])/self.DT
            self.initial_v = math.sqrt(self.initial_vx**2 + self.initial_vy**2)
        else:
            self.initial_v = 0.0
        self.initial_state = State(x=self.cx[0], y=self.cy[0], yaw=self.cyaw[0], v=self.initial_v/2)
        self.goal = [self.cx[-1], self.cy[-1]]
        self.state = self.initial_state

        # initial yaw compensation
        if self.state.yaw - self.cyaw[0] >= math.pi:
            self.state.yaw -= math.pi * 2.0
        elif self.state.yaw - self.cyaw[0] <= -math.pi:
            self.state.yaw += math.pi * 2.0

        self.time = 0.0
        self.x = [self.state.x]
        self.y = [self.state.y]
        self.yaw = [self.state.yaw]
        self.v = [self.state.v]
        self.vel_x = [self.state.v * math.cos(self.state.yaw)]
        self.vel_y = [self.state.v * math.sin(self.state.yaw)]
        self.vel_yaw = [0]
        self.t = [0.0]
        self.d = [0.0]
        self.a = [0.0]
        
        self.target_ind, _ = self.calc_nearest_index(self.state, self.cx, self.cy, self.cyaw, 0)
        self.odelta, self.oa = None, None
        self.ai, self.di = 0, 0

        self.index = 0
        self.reached_goal = 0

        self.state_future_vel_yaw = []
        self.state_future_velocity_x = []
        self.state_future_velocity_y = []

        self.main_car_path_index = 0
        self.vel_yaw_cache = 0
        

    def update(self, obs_cache):
        if(self.time < self.MAX_TIME):
            if(math.sqrt((self.state.x - self.goal[0])**2+(self.state.y - self.goal[1])**2) < self.XY_GOAL_TOLERANCE):
                return 1

            # 记录数据
            if(self.di == 0):
                self.state_future_vel_yaw.append(0)
            else:
                self.state_future_vel_yaw.append(self.state.v/self.average_length/math.tan(self.di))
            self.state_future_velocity_x.append(self.state.v*math.cos(self.state.yaw))
            self.state_future_velocity_y.append(self.state.v*math.sin(self.state.yaw))
            
            # 更新MPC
            try:
                self.xref, self.target_ind, self.dref, self.xref2, self.dref2 = self.calc_ref_trajectory(
                    self.state, self.cx, self.cy, self.cyaw, self.ck, self.sp, self.DL, self.target_ind)

                self.x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]  # current state

                self.oa, self.odelta, self.ox, self.oy, self.oyaw, self.ov = self.iterative_linear_mpc_control(
                    self.xref, self.x0, self.dref, self.oa, self.odelta, self.xref2, self.dref2)

                if self.odelta is not None:
                    self.di, self.ai = self.odelta[0], self.oa[0]
            except:
                self.di, self.ai = 0, 0
            
            if(self.OBSTACLE_AVOIDANCE):
                # 计算障碍物距离以及方向单位向量
                self.distances = []
                self.orientations = []
                for self.obstacle in obs_cache:
                    self.obs_x = self.obstacle[0]
                    self.obs_y = self.obstacle[1]
                    self.distance = math.sqrt((self.state.x - self.obs_x)**2+(self.state.y - self.obs_y)**2)
                    self.distances.append(self.distance)
                    self.orientations.append([(self.obs_x - self.state.x)/self.distance, (self.obs_y - self.state.y)/self.distance])
                
                # 人工势场法被动避障矢量方向
                self.force = [0, 0]
                for self.dd in range(len(self.distances)):
                    temp_psi = self.pi_2_pi(np.arctan2(self.orientations[self.dd][1], self.orientations[self.dd][0]))
                    temp_car_psi = self.pi_2_pi(self.state.yaw)
                    temp_delta_psi = self.pi_2_pi(temp_psi - temp_car_psi)
                    # 只考虑直行方向上前后夹角30度范围内的障碍物
                    dangle = 0.52
                    if(abs(temp_delta_psi)<=dangle or abs(temp_delta_psi)>=math.pi-dangle):
                        self.distance = self.distances[self.dd]
                        self.vv = self.calc_v(self.distance, self.state.v)
                        self.force[0] += self.vv*self.orientations[self.dd][0]
                        self.force[1] += self.vv*self.orientations[self.dd][1]
                
                # 如果未到达终点
                if(self.reached_goal == 0):
                    # 通过人工势场法进行被动避障
                    if(self.force[0] != 0 or self.force[1] != 0):
                        self.pf_vx = -self.force[0]
                        self.pf_vy = -self.force[1]
                        self.mpc_vx = (self.state.v+self.ai*self.DT)*math.cos(self.state.yaw)
                        self.mpc_vy = (self.state.v+self.ai*self.DT)*math.sin(self.state.yaw)
                        self.vx = self.pf_vx + self.mpc_vx
                        self.vy = self.pf_vy + self.mpc_vy
                        if(self.SHOW_ANIMATION and self.SHOW_POTENTIAL_FIELD):
                            plt.plot([self.state.x, self.state.x+self.vx], [self.state.y, self.state.y+self.vy],c='g')
                        self.vv = math.sqrt(self.vx**2 + self.vy**2)
                        self.psi = self.pi_2_pi(np.arctan2(self.vy, self.vx))
                        self.car_psi = self.pi_2_pi(self.state.yaw)
                        self.delta_psi = self.pi_2_pi(self.psi - self.car_psi)
                        self.u0 = self.vv*math.cos(self.delta_psi)
                        self.di_part = 0.5
                        self.u1 = (self.di*self.di_part + self.delta_psi*(1-self.di_part))*2
                        #ai = (u0-state.v)/DT
                        self.ai_part = 0.9
                        self.ai = self.ai*self.ai_part + (self.u0-self.state.v)/self.DT*(1-self.ai_part)
                        if(self.ai<0):
                            self.u1 = -self.u1
                        #self.di = self.u1

                    # 根据MPC解算的控制进行运动
                    else:
                        self.ai = self.ai
                        self.di = self.di
                # 如果已经到达终点
                else:
                    # 停车
                    self.ai = -self.state.v/self.DT
                    self.di = 0
            
                self.state = self.update_state(self.state, self.ai, self.di)
                self.time = self.time + self.DT

                self.x.append(self.state.x)
                self.y.append(self.state.y)
                self.yaw.append(self.state.yaw)
                self.v.append(self.state.v)
                self.vel_x.append(self.state.v * math.cos(self.state.yaw))
                self.vel_y.append(self.state.v * math.sin(self.state.yaw))
                if(self.di == 0):
                    self.vel_yaw.append(0)
                else:
                    self.vel_yaw.append(self.state.v/self.average_length/math.tan(self.di))
                self.t.append(self.time)
                self.d.append(self.di)
                self.a.append(self.ai)

                if self.check_goal(self.state, self.goal, self.target_ind, len(self.cx)):
                    self.reached_goal = 1
                    #print("Goal")
                
                if(math.sqrt((self.state.x - self.goal[0])**2+(self.state.y - self.goal[1])**2) < self.XY_GOAL_TOLERANCE):
                    self.reached_goal = 1
                    #print("Goal")

                if self.SHOW_ANIMATION:  # pragma: no cover
                    plt.plot(self.cx, self.cy, "-r", label="course")
                    plt.plot(self.x, self.y, c='b', label="trajectory")
                    plt.plot(self.xref[0, :], self.xref[1, :], "xk", label="xref")
                    plt.plot(self.xref2[0, :], self.xref2[1, :], "xk", label="xref2")
                    plt.plot(self.cx[self.target_ind], self.cy[self.target_ind], "xg", label="target")
                return 0

            # 如果是主车
            else:
                # 确保索引未超出waypoints列表
                if(self.main_car_path_index<len(self.cx)):
                    if(self.main_car_path_index<len(self.cx)-2):
                        self.state.x = self.cx[self.main_car_path_index+1]
                        self.state.y = self.cy[self.main_car_path_index+1]
                        self.state.yaw = math.atan2(self.cy[self.main_car_path_index+2]-self.cy[self.main_car_path_index+1], self.cx[self.main_car_path_index+2]-self.cx[self.main_car_path_index+1])
                        self.state.v = math.sqrt((self.cx[self.main_car_path_index+1]-self.cx[self.main_car_path_index])**2 + (self.cy[self.main_car_path_index+1]-self.cy[self.main_car_path_index])**2)/self.DT
                    elif(self.main_car_path_index==len(self.cx)-2):
                        self.state.x = self.cx[self.main_car_path_index+1]
                        self.state.y = self.cy[self.main_car_path_index+1]
                        self.state.yaw = math.atan2(self.cy[self.main_car_path_index+1]-self.cy[self.main_car_path_index], self.cx[self.main_car_path_index+1]-self.cx[self.main_car_path_index])
                        self.state.v = math.sqrt((self.cx[self.main_car_path_index+1]-self.cx[self.main_car_path_index])**2 + (self.cy[self.main_car_path_index+1]-self.cy[self.main_car_path_index])**2)/self.DT
                    else:
                        self.state.x = self.cx[self.main_car_path_index]
                        self.state.y = self.cy[self.main_car_path_index]
                        self.state.yaw = math.atan2(self.cy[self.main_car_path_index]-self.cy[self.main_car_path_index-1], self.cx[self.main_car_path_index]-self.cx[self.main_car_path_index-1])
                        self.state.v = math.sqrt((self.cx[self.main_car_path_index]-self.cx[self.main_car_path_index-1])**2 + (self.cy[self.main_car_path_index]-self.cy[self.main_car_path_index-1])**2)/self.DT

                    self.x.append(self.state.x)
                    self.y.append(self.state.y)
                    self.yaw.append(self.state.yaw)
                    self.v.append(self.state.v)
                    self.vel_x.append(self.state.v * math.cos(self.state.yaw))
                    self.vel_y.append(self.state.v * math.sin(self.state.yaw))
                    self.vel_yaw.append((self.state.yaw-self.vel_yaw_cache)/self.DT)

                    self.main_car_path_index += 1
                    self. vel_yaw_cache = self.state.yaw
                else:
                    self.reached_goal = 1

                if self.SHOW_ANIMATION:  # pragma: no cover
                    plt.plot(self.cx, self.cy, "-r", label="course")
                    plt.plot(self.x, self.y, c='b', label="trajectory")
                
                return 0
        else:
            return 1

if __name__ == '__main__':
    # 读取数据
    data = pickle.load(open(r'sample.pickle','rb'))
    wp_length = data['state/future/x'].shape[1]
    print('mpc input waypoints length: '+str(wp_length))
    # 读取轨迹数量
    car_num = len(data['state/id'])
    print('num of cars: '+str(car_num))
    # 设定选中的汽车
    car_index = 15
    # 生成单车辆MPC控制器
    car = SINGLE_MPC()
    # 初始化轨迹
    car.setup(data, car_index)

    # 开始仿真
    obstacles = []
    ticks = 0
    break_flag = 1
    while(break_flag == 1):
        if(car.SHOW_ANIMATION):
            plt.clf()
            car.plot_car(car.state.x, car.state.y, car.state.yaw, steer=car.di)
        reached = car.update(obstacles)
        if(car.SHOW_ANIMATION):
            plt.pause(0.001)
        if(reached):
            break_flag = 0