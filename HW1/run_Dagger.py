#!/usr/bin/env python
# this file runs the BC algorithm for learning from expert policy
# python run_BC.py expert_data/Humanoid-v2.pkl experts/Humanoid-v2.pkl Humanoid-v2 --render \ --num_roll_outs 20

# importing important libraries
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
import pickle as pkl
import tensorflow as tf
import tf_util
import gym
import load_policy


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pkl.loads(f.read())
    return data


def train_model(expert_data,env_name):
    # this function used for training the model

    obs_data = expert_data['observations']
    obs_data = np.reshape(obs_data, (obs_data.shape[0], obs_data.shape[1]))
    act_data = expert_data['actions']
    act_data = np.reshape(act_data, (act_data.shape[0], act_data.shape[2]))

    # defining model for the behavioral cloning
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=obs_data.shape[1]))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(act_data.shape[1], activation='linear'))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=[ 'accuracy' ])
    model.fit(obs_data, act_data, epochs=100, batch_size=128, shuffle=True)
    model.save('models/' + env_name + '_cloned_model.h5')



# simulate the data and save it into appropriate file
def simulate_dagger(policy_fn, env_name, num_roll_outs, render):

    with tf.Session():
        tf_util.initialize()
        model = load_model('models/' + env_name + '_cloned_model.h5')
        env = gym.make(env_name)
        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_roll_outs):
            print('iter', i)
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            while not done:
                # resizing the observation vector for the model prediction
                obs_model = np.array(obs)
                obs_model = np.reshape(obs_model, (1, obs_model.shape[0]))
                observations.append(obs)

                # predicting appropriate action with the model and withe policy
                # saving the action of the policy and stepping with the model
                model_action = model.predict(obs_model)
                expert_action = policy_fn(obs[None, :])
                actions.append(expert_action)
                obs, r, done, _ = env.step(model_action)
                total_reward += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(total_reward)

        # print the returns and save the new data in dictionary each is a numpy array
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        expert_new_data = {'observations': np.array(observations), 'actions': np.array(actions)}

        cache_returns = (returns, np.mean(returns), np.std(returns))

        return cache_returns, expert_new_data


def dagger_step(expert_data, policy_fn, env_name, num_roll_outs, render):
    train_model(expert_data, env_name)
    cache_returns, expert_new_data = simulate_dagger(policy_fn, env_name, num_roll_outs, render)

    return cache_returns, expert_new_data


def main():
    # importing settings from the command window
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('env_name', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_time_steps', type=int)
    parser.add_argument('--num_roll_outs', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()
    std_r = []
    mean_r = []

    # assigning those settings to variables

    expert_data_file = args.expert_data_file
    expert_policy_file = args.expert_policy_file
    env_name = args.env_name
    num_roll_outs = args.num_roll_outs
    render = args.render

    # extracting expert data and policy and adjust it's size from pickle file
    expert_data = load_data(expert_data_file)
    policy_fn = load_policy.load_policy(expert_policy_file)
    for i in range(20):
        cache_returns, expert_new_data = dagger_step(expert_data, policy_fn, env_name, num_roll_outs, render)
        obs_data = expert_new_data['observations']
        act_data = expert_new_data['actions']
        expert_obs_data = np.append(expert_data['observations'], obs_data, axis=0)
        expert_act_data = np.append(expert_data['actions'], act_data, axis=0)
        expert_data = {'observations': expert_obs_data, 'actions': expert_act_data}
        # print(expert_data['observations'].shape)
        # print(expert_data['actions'].shape)
        std_r.append(cache_returns[2])
        mean_r.append(cache_returns[1])
    print(std_r)
    print(mean_r)


if __name__ == "__main__":
    main()
