import argparse
import os
from itertools import count


import os
import logging
import abc
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork,ActorNetwork,CriticNetwork
from visualizer import Visualizer
from replay_buffer import ReplayBuffer
'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''





class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None,
                 chart_data=None, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05,
                 net='dnn', num_steps=1, lr=0.001,
                 value_network=None, policy_network=None,
                 output_path='', reuse_models=True):
        # 인자 확인
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        # 강화학습 기법 설정
        self.rl_method = rl_method
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        self.critic=value_network
        self.actor=policy_network
        self.tau=0.001
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_value2 = []
        self.memory_target_policy = []
        self.memory_target_value = []
        self.memory_target_action = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        # 로그 등 출력 경로
        self.output_path = output_path
        # for Delayed Policy Update
        self._update_step = 0
        self._target_update_interval = 2

    def init_policy_network(self, shared_network=None,
                            activation='sigmoid', loss='binary_crossentropy'):
        if self.rl_method == 'td3':
            print("actor")
            self.actor = ActorNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,num_steps=self.num_steps, activation=activation, loss=loss
                , lr=self.lr)
            print(self.actor)
        elif self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and \
                os.path.exists(self.policy_network_path):
            self.policy_network.load_model(
                model_path=self.policy_network_path)

    def init_value_network(self, shared_network=None,
                           activation='linear', loss='mse'):
        if self.rl_method =='td3':
            self.critic = CriticNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,num_steps=self.num_steps, activation=activation, loss=loss,
                lr=self.lr)
        elif self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and \
                os.path.exists(self.value_network_path):
            self.value_network.load_model(
                model_path=self.value_network_path)


    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_target_policy=[]
        self.memory_target_value = []
        self.memory_target_action=[]
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_value2 = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[
                self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    @abc.abstractmethod
    def train(self, batch_size, delayed_reward, discount_factor):
        pass

    def update_networks(self,
                        batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        x, policy,y_value1,y_value2,critic_target = self.get_batch(
            batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            loss = 0
            loss += self.critic.train(x, y_value2, y_value2, critic_target)
            if self._update_step % self._target_update_interval == 0:
                # update actor
                loss +=  self.actor.train(x,policy)

                # update target networks
                self.actor.target_update()
                self.critic.target_update()
            self._update_step = self._update_step + 1 #reset 까먹지 않기

            return loss
        return None

    def fit(self, delayed_reward, discount_factor, full=False):
        batch_size = len(self.memory_reward) if full \
            else self.batch_size
        # 배치 학습 데이터 생성 및 신경망 갱신
        if batch_size > 0:
            _loss = self.update_networks(
                batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] \
                             * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) \
                                 + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] \
                                          * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                                + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] \
                                           * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                                 + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] \
                         * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches,
            epsilon=epsilon, action_list=Agent.ACTIONS,
            actions=self.memory_action,
            num_stocks=self.memory_num_stocks,
            outvals_value=self.memory_value,
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx,
            learning_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance,
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(
            self.epoch_summary_dir,
            'epoch_summary_{}.png'.format(epoch_str))
        )

    def run(
            self, num_epoches=100, balance=10000000,
            discount_factor=0.9, start_epsilon=0.5, learning=True):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr} " \
               "DF:{discount_factor} TU:[{min_trading_unit}," \
               "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        )
        with self.lock:
            logging.info(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(
            self.output_path, 'epoch_summary_{}'.format(
                self.stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)

            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()
            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon \
                          * (1. - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon
            while True:
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_value2 = None
                pred_policy = None
                pred_target_policy=None
                pred_target_value = None
                if self.critic is not None:
                   pred_value = self.critic.predict(
                   list(q_sample))
                   pred_value2 = self.critic.predict2(
                       list(q_sample))
                if self.actor is not None:
                    pred_policy = self.actor.predict(
                        list(q_sample))
                    pred_target_policy =self.actor.target_model1_predict(list(q_sample))

                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = \
                    self.agent.decide_action(pred_value, pred_policy, epsilon)

                # target 값을 이용한 행동 결정
                target_action, target_confidence, target_exploration = \
                    self.agent.decide_action(pred_target_policy, pred_target_value, epsilon)

                #결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = \
                    self.agent.act(action, confidence)


                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                self.memory_target_action.append(target_action)
                self.memory_target_policy.append(pred_target_policy)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                    self.memory_value2.append(pred_value2)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

                # 지연 보상 발생된 경우 미니 배치 학습
                if learning and (delayed_reward != 0):
                    self.fit(delayed_reward, discount_factor)
            # 에포크 종료 후 학습
            if learning:
                self.fit(
                    self.agent.profitloss, discount_factor, full=True)
            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                             "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                             "#Stocks:{} PV:{:,.0f} "
                             "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                    self.stock_code, epoch_str, num_epoches, epsilon,
                    self.exploration_cnt, self.itr_cnt,
                    self.agent.num_buy, self.agent.num_sell,
                    self.agent.num_hold, self.agent.num_stocks,
                    self.agent.portfolio_value, self.learning_cnt,
                    self.loss, elapsed_time_epoch))

            # 에포크 관련 정보 가시화
            self.visualize(epoch_str, num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logging.info("[{code}] Elapsed Time:{elapsed_time:.4f} "
                         "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                code=self.stock_code, elapsed_time=elapsed_time,
                max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))

    def save_models(self):
        if self.value_network is not None and \
                self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and \
                self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)
REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99


class TD3(ReinforcementLearner):
    """docstring for TD3"""

    def __init__(self, *args, shared_network=None,
                 value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'td3'  # name for uploading results
        if shared_network is None:
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps,
                input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=shared_network)

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
            reversed(self.memory_target_policy[-batch_size:]),
            reversed(self.memory_target_action[-batch_size:]),
            reversed(self.memory_value2[-batch_size:])
        )
        sample_batch = np.zeros((batch_size, self.num_steps, self.num_features))
        x_sample_next = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_value2 = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        y_target_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_target_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        rewards=np.zeros(batch_size)
        action_batch = np.asarray([data[1] for data in memory])
        value_max_next = 0
        value_max_next2 = 0
        target_max_next = 0
        x_sample_next[0] = self.memory_sample[-1]
        reward_next = self.memory_reward[-1]

        for i, (sample, action, value, policy, reward,target_policy,target_action, value2) \
                in enumerate(memory):
              sample_batch[i] = sample
              y_value[i] = value
              y_value2[i] = value2
              y_policy[i] = policy
              q1_vals = self.critic.target_model1_predict(sample)
              q2_vals = self.critic.target_model2.predict(sample)
              y_target_value[i] = np.min(np.vstack([q1_vals.transpose(),q2_vals.transpose()]),axis=0)
              y_target_policy[i] = target_policy
              rewards[i] = (delayed_reward + reward_next - reward * 2) * 100
              y_target_value[i, target_action] = rewards[i] + discount_factor * target_max_next # q_value
        #     # y_target_policy[i, target_action] = sigmoid(target_value[target_action])
              y_value[i, action] = value_max_next # q_value
              y_value2[i, action] = value_max_next2  # q_value
              y_policy[i, action] = sigmoid(y_value[action])
              target_max_next = y_target_value.max()
              value_max_next = value.max()
              value_max_next2 = value2.max()
              reward_next = reward





        return sample_batch, y_policy, y_value,y_value2, y_target_value


