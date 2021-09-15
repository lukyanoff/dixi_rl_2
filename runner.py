from collections import deque
import numpy as np


def train_agent(env, agent, visualize=False, train_episodes=99999, training_batch_size=500):
    #agent.create_writer(env.initial_balance, env.normalize_value, train_episodes)  # create TensorBoard writer
    total_average = deque(maxlen=20)  # save recent 100 episodes net worth
    best_average_reward = 0  # used to track best average net worth

    for episode in range(train_episodes):
        #state = env.reset(env_steps_size=training_batch_size)
        state = env.reset()
        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        while True:
            #env.render(visualize)

            action, prediction = agent.act(state)

            next_state, reward, done, info = env.step(action)
            # print("Step ", info)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(2)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state
            if done:
                break
        total_average.append(reward)
        print("Reward", reward, np.average(total_average))
        # print("Reward meta", info)

        a_loss, c_loss = agent.replay(states, actions, rewards, predictions, dones, next_states)
        # total_average.append(env.net_worth)
        # average = np.average(total_average)

        # agent.writer.add_scalar('Data/average net_worth', average, episode)
        # agent.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)

        # print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(episode, env.net_worth, average, env.episode_orders))
        # if episode > len(total_average):
        #     if best_average_reward < average:
        #         best_average = average
        #         print("Saving model")
        #         agent.save(score="{:.2f}".format(best_average), args=[episode, average, env.episode_orders, a_loss, c_loss])
        #     agent.save()