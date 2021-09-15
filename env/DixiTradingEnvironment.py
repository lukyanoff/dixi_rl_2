from collections import deque

import numpy as np
import pandas as pd
from gym import Env, spaces


Action_NO = 0
Action_YES = 1

OneStepStateItems_DOWN = 0
OneStepStateItems_UP = 1


class DixiTradingEnvironment(Env):
    def __init__(self, df: pd.DataFrame, initial_position: int, window_size:int):
        super().__init__()

        self.initial_position = initial_position
        self.window_size = window_size

        self.commision_fee = 0#0.8

        self._data = df.to_dict('list')
        self._data['action'] = np.zeros(len(self._data['close']))


        self.action_space = spaces.Discrete(2)

        # set observation_space
        columns = self._data.keys()
        self.observation_space = spaces.Space(shape=(len(columns), window_size))

        self.step_index = Action_NO
        self.reset()


    def step(self, action):
        self._data['action'][self.step_index] = action
        self.step_index = self.step_index + 1

        done = self.done()
        if not done:
            reward, rewards_metadata = 0, {}
        else:
            reward, rewards_metadata = self._get_reward()

        info = {
            'step': self.step_index,
            'rewards_metadata': rewards_metadata
        }

        current_state = self.get_current_state()
        return current_state, reward, done, info

    def done(self):

        generator_reached_the_end = self.step_index >=  len(self._data['close']) - 1 # we need one step for closing latest position
        return generator_reached_the_end

    def _get_reward(self):
        START_PENALTY_AFTER_N_STEPS = 5
        LAZY_PENALTY = 0.01

        if self.step_index == 0:
            return 0

        closed_trades = []
        current_open_trade = None
        for i in range(self.initial_position, self.step_index):
            position_action = self._data['action'][i]
            prev_position_action = self._data['action'][i-1]

            if position_action == Action_YES and prev_position_action == Action_NO:
                current_open_trade = self._data['close'][i]

            if position_action == Action_NO and prev_position_action == Action_YES:
                closed_trades.append({
                    'open': current_open_trade,
                    'close': self._data['close'][i]
                })
                current_open_trade = None

        # close last position
        if current_open_trade is not None:
            closed_trades.append({
                'open': current_open_trade,
                'close': self._data['close'][-1]
            })
            current_open_trade = None

        #calculate total reward
        total_reward = 0
        rewards = []
        for trade in closed_trades:
            open_price = trade['open']
            close_price = trade['close']
            absolute_profit_per_share = close_price - open_price

            absolute_profit = absolute_profit_per_share# * 100 # AMOUNT_OF_SHARES
            rewards.append(absolute_profit)
            total_reward = total_reward + absolute_profit# - self.commision_fee
        return total_reward, {
            'rewards': rewards
        }

    def get_current_state(self):
        result = []
        for key in self._data.keys():
            arr = self._data[key][self.step_index - self.window_size : self.step_index]
            result.append(arr)
        return result

    def _reset(self):
        self.step_index = self.initial_position
        self._data['action'] = np.zeros(len(self._data['close']))

    def reset(self):
        self._reset()
        return self.get_current_state()


    def render(self):
        # self._get_state()
        pass
