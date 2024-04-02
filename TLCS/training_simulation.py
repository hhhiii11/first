import traci
import numpy as np
import random
import timeit
import os
from collections import Counter

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Model1, Model2, Memory1, Memory2, Memory3, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Model1 = Model1
        self._Model2 = Model2
        self._Memory1 = Memory1
        self._Memory2 = Memory2
        self._Memory3 = Memory3
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._pre_cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._pre_avg_queue_length_store = []
        self._co_emi_store = []
        self._pre_co_emi_store_store = []
        self._training_epochs = training_epochs


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # 生成道路文件并激活sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        self._co_emi = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._pre_sum_queue_length = 0
        self._sum_waiting_time = 0
        self._pre_sum_waiting_time = 0
        self._sum_co_emi = 0
        self._pre_sum_co_emi = 0

        old_total_wait = 0
        old_total_que = 0
        old_total_co = 0


        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # 获取当前状态current_state
            current_state = self._get_state()

            # 计算上一步操作后的累计等待时间current_total_wait并计算奖励reward（操作前后的累计等待时间差）
            # waiting time = 所有汽车进入环境之后累计等待的秒数，通过定义的_collect_waiting_times方法获取获取
            # old_total_wait初始为0，每次模拟步结束后将当前等待时间赋给old_total_wait作为操作前的累计等待时间
            current_total_wait = self._collect_waiting_times()
            current_total_que = self._collect_que_len()
            current_total_co = self._collect_co_emi()

            reward = old_total_wait - current_total_wait
            reward1 = old_total_que - current_total_que
            reward2 = old_total_co - current_total_co

            # 将上一步操作后的结果作为经验存入memory
            # ！！！构建多个相互独立的memory
            if self._step != 0:
                self._Memory1.add_sample((old_state, old_action, reward, current_state))
                self._Memory2.add_sample((old_state, old_action, reward1, current_state))
                self._Memory3.add_sample((old_state, old_action, reward2, current_state))

            # 根据当前状态current_state选择激活的信号相位
            action = self._choose_action(current_state, epsilon)

            # 如果所选相位与上一阶段不同，先激活上一个绿灯相位的黄色相位
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # 执行刚刚选择的动作切换相位
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # 保存本次仿真的初始状态和选择的动作以及累计等待时间
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            old_total_que = current_total_que
            old_total_co = current_total_co

            # 仅保存有意义的经验以便代理更好的学习
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
            self._replay1()
            self._replay2()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # 如果执行完切换相位后会超出最大时间则不执行超出部分的时长
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # 在sumo中进行一步仿真（1s）
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            pre_queue_length = self._pre_get_queue_length()
            co_emi = self._get_co_emi()
            pre_co_emi = self._pre_get_co_emi()
            self._sum_co_emi += co_emi #每一步仿真1秒，因此排放量就等于每秒的排放速率co_emi累加
            self._pre_sum_co_emi += pre_co_emi
            self._pre_sum_queue_length += pre_queue_length
            self._sum_waiting_time += queue_length
            self._sum_queue_length += queue_length #排队时1步意味着每辆车等待1秒，因此queue_lenght = waited_seconds
            self._pre_sum_waiting_time += pre_queue_length #排队时1步意味着每辆车等待1秒，因此queue_lenght = waited_seconds

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            vehicle_type = traci.vehicle.getTypeID(car_id)
            if vehicle_type == "emergency_car":
                vehicle_priority = 50
            else:
                vehicle_priority = 1
            road_id = traci.vehicle.getRoadID(car_id)  # 获取汽车所在的道路 ID
            if road_id in incoming_roads:  # 只考虑汽车在进车道路上的等待时间
                self._waiting_times[car_id] = wait_time*vehicle_priority
            else:
                if car_id in self._waiting_times:  # 如果被统计的车辆已经离开路口则删除
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _collect_co_emi(self):
        """
        co2排放速率
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            co_emi = traci.vehicle.getCO2Emission(car_id)
            vehicle_type = traci.vehicle.getTypeID(car_id)
            if vehicle_type == "emergency_car":
                vehicle_priority = 50
            else:
                vehicle_priority = 1
            road_id = traci.vehicle.getRoadID(car_id)  # 获取汽车所在的道路 ID
            if road_id in incoming_roads:  # 只考虑汽车在进车道路上的排放速率
                self._co_emi[car_id] = co_emi*vehicle_priority
            else:
                if car_id in self._co_emi:  # 如果被统计的车辆已经离开路口则删除
                    del self._co_emi[car_id]
        total_co_emi = sum(self._co_emi.values())
        return total_co_emi

    def _collect_que_len(self):
        """
        Retrieve the que_len of every car in the incoming roads
        """

        total_que_len = self._get_queue_length()
        return total_que_len

    def _choose_action(self, state, epsilon):
        """
        根据 epsilon 贪婪策略决定执行探索性或最优操作
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # 随机动作
        else:
            n = np.argmax(self._Model.predict_one(state))
            n1 = np.argmax(self._Model1.predict_one(state))
            n2 = np.argmax(self._Model2.predict_one(state))
            lst = [n, n1, n2]
            # 使用Counter函数统计每个元素出现的次数
            cnt = Counter(lst)
            # 找到出现次数最多的元素
            most_common = max(cnt, key=cnt.get)
            return most_common  # 最优策略

    def _set_yellow_phase(self, old_action):
        """
        激活黄灯相位
        """
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        激活绿灯相位
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        检索每个进车道中速度 = 0 的汽车数量
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _pre_get_queue_length(self):
        """
        检索每个进车道中速度 = 0 的优先车辆数量
        """
        halt_N = 0
        halt_S = 0
        halt_E = 0
        halt_W = 0
        for vehicleID in traci.edge.getLastStepVehicleIDs("N2TL"):
            if traci.vehicle.getTypeID(vehicleID) == "emergency_car" and traci.vehicle.getSpeed(
                    vehicleID) <= 0.1 and traci.vehicle.getLanePosition(vehicleID) > 0:
                halt_N += 1
        for vehicleID in traci.edge.getLastStepVehicleIDs("S2TL"):
            if traci.vehicle.getTypeID(vehicleID) == "emergency_car" and traci.vehicle.getSpeed(
                    vehicleID) <= 0.1 and traci.vehicle.getLanePosition(vehicleID) > 0:
                halt_S += 1
        for vehicleID in traci.edge.getLastStepVehicleIDs("E2TL"):
            if traci.vehicle.getTypeID(vehicleID) == "emergency_car" and traci.vehicle.getSpeed(
                    vehicleID) <= 0.1 and traci.vehicle.getLanePosition(vehicleID) > 0:
                halt_E += 1
        for vehicleID in traci.edge.getLastStepVehicleIDs("W2TL"):
            if traci.vehicle.getTypeID(vehicleID) == "emergency_car" and traci.vehicle.getSpeed(
                    vehicleID) <= 0.1 and traci.vehicle.getLanePosition(vehicleID) > 0:
                halt_W += 1
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _pre_get_co_emi(self):
        """
        获取优先车辆co2排放速率
        """
        pre_co_emi = 0
        for vehicleID in traci.edge.getLastStepVehicleIDs("N2TL"):
            if traci.vehicle.getTypeID(vehicleID) == "emergency_car":
                pre_co_emi += traci.vehicle.getCO2Emission(vehicleID)
        for vehicleID in traci.edge.getLastStepVehicleIDs("S2TL"):
            if traci.vehicle.getTypeID(vehicleID) == "emergency_car":
                pre_co_emi += traci.vehicle.getCO2Emission(vehicleID)
        for vehicleID in traci.edge.getLastStepVehicleIDs("E2TL"):
            if traci.vehicle.getTypeID(vehicleID) == "emergency_car":
                pre_co_emi += traci.vehicle.getCO2Emission(vehicleID)
        for vehicleID in traci.edge.getLastStepVehicleIDs("W2TL"):
            if traci.vehicle.getTypeID(vehicleID) == "emergency_car":
                pre_co_emi += traci.vehicle.getCO2Emission(vehicleID)
        return pre_co_emi

    def _get_co_emi(self):
        """
        获取所有车辆co2排放速率
        """
        co_emi = 0
        for vehicleID in traci.edge.getLastStepVehicleIDs("N2TL"):
                co_emi += traci.vehicle.getCO2Emission(vehicleID)
        for vehicleID in traci.edge.getLastStepVehicleIDs("S2TL"):
                co_emi += traci.vehicle.getCO2Emission(vehicleID)
        for vehicleID in traci.edge.getLastStepVehicleIDs("E2TL"):
                co_emi += traci.vehicle.getCO2Emission(vehicleID)
        for vehicleID in traci.edge.getLastStepVehicleIDs("W2TL"):
                co_emi += traci.vehicle.getCO2Emission(vehicleID)
        return co_emi

    def _get_state(self):
        """
        以单元占用的形式从sumo中检索十字路口的状态
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:

            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            speed = traci.vehicle.getSpeed(car_id)
            vehicle_type = traci.vehicle.getTypeID(car_id)
            if vehicle_type == "emergency_car":
                vehicle_priority = 50
            else:
                vehicle_priority = 1
            lane_pos = 750 - lane_pos

            # 将车道根据距离信号灯的距离划分
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # 查找汽车所在的车道
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False

            if valid_car:
                if state[car_position] == 50 or state[car_position] == 100:
                    pass
                else:
                    state[car_position] = vehicle_priority
                    state[car_position+80] = speed
        return state


    def _replay(self):
        """
        从memory中检索一组样本，并为每个样本更新学习方程，然后训练
        """
        batch = self._Memory1.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # 经验池是否填满
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])

            q_s_a = self._Model.predict_batch(states)  # 预测 Q(state)
            q_s_a_d = self._Model.predict_batch(next_states)  # 预测 Q(next_state)

            # 设置训练队列
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # 从一个样本中提取数据
                current_q = q_s_a[i]  # 获取之前预测的Q(state)
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # 更新 Q(state, action)
                x[i] = state
                y[i] = current_q

            self._Model.train_batch(x, y)  # train CNN
            self._Model.train_target(x, y) 

    def _replay1(self):
        """
        从memory中检索一组样本，并为每个样本更新学习方程，然后训练
        """
        batch = self._Memory2.get_samples(self._Model1.batch_size)

        if len(batch) > 0:  # 经验池是否填满
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])

            q_s_a = self._Model1.predict_batch(states)  # 预测 Q(state)
            q_s_a_d = self._Model1.predict_batch(next_states)  # 预测 Q(next_state)

            # 设置训练队列
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # 从一个样本中提取数据
                current_q = q_s_a[i]  # 获取之前预测的Q(state)
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # 更新 Q(state, action)
                x[i] = state
                y[i] = current_q

            self._Model1.train_batch(x, y)  # train CNN
            self._Model1.train_target(x, y)

    def _replay2(self):
        """
        从memory中检索一组样本，并为每个样本更新学习方程，然后训练
        """
        batch = self._Memory3.get_samples(self._Model2.batch_size)

        if len(batch) > 0:  # 经验池是否填满
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])

            q_s_a = self._Model2.predict_batch(states)  # 预测 Q(state)
            q_s_a_d = self._Model2.predict_batch(next_states)  # 预测 Q(next_state)

            # 设置训练队列
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # 从一个样本中提取数据
                current_q = q_s_a[i]  # 获取之前预测的Q(state)
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # 更新 Q(state, action)
                x[i] = state
                y[i] = current_q

            self._Model2.train_batch(x, y)  # train CNN
            self._Model2.train_target(x, y)

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._pre_cumulative_wait_store.append(self._pre_sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)
        self._pre_avg_queue_length_store.append(self._pre_sum_queue_length / self._max_steps)
        self._co_emi_store.append(self._sum_co_emi)
        self._pre_co_emi_store_store.append(self._pre_sum_co_emi)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def pre_cumulative_wait_store(self):
        return self._pre_cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

    @property
    def pre_avg_queue_length_store(self):
        return self._pre_avg_queue_length_store\

    @property
    def co_emi_store(self):
        return self._co_emi_store

    @property
    def pre_co_emi_store(self):
        return self._pre_co_emi_store_store

