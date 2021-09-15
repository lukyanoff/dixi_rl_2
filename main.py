import pandas as pd
from agent.DixiAgent import DixiAgent
from env.DixiTradingEnvironment import DixiTradingEnvironment
import runner as helper



def train_agent():
    train_df = pd.read_csv("NYSE:BABA_dataset.csv")
    train_df.set_index('date', inplace=True, drop=False)

    train_df.drop(columns=['date', 'volume', 'open', 'low'], inplace=True)
    train_df.info()

    trading_environment = DixiTradingEnvironment(df=train_df, initial_position=100, window_size=5)
    state = trading_environment.reset()
    print("Expected state shape", trading_environment.observation_space.shape)
    print("Real state shape", state)

    print("Action space shape", trading_environment.action_space.shape, trading_environment.action_space.sample())


    trading_agent = DixiAgent(environment=trading_environment)
    helper.train_agent(trading_environment, trading_agent, visualize=False, train_episodes=50000, training_batch_size=500)


def test_environment():
    from datetime import datetime

    train_df = pd.read_csv("NYSE:BABA_dataset.csv")
    train_df.set_index('date', inplace=True, drop=False)
    # # train_df.reset_index(drop=True, inplace=True)
    # train_df.to_csv("NYSE:BABA_dataset.csv", index=False)


    train_df.drop(columns=['date', 'volume', 'open', 'low'], inplace=True)
    train_df.info()
    env = DixiTradingEnvironment(df=train_df, initial_position=100, window_size=5)

    for i in range(5):
        start_time = datetime.now()
        obs = env.reset()
        iteration_count = 0
        while True:
            iteration_count = iteration_count + 1
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            if done:
                print("REWARD", reward)
                env.reset()
                break
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        print(f"{iteration_count} iteration_count {elapsed_time} time")

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    # tf.debugging.set_log_device_placement(True)
    # with tf.device('/GPU:0'):
    train_agent()

