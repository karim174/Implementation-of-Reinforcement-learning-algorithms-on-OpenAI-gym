#!/usr/bin/env python
# this file runs the BC algorithm for learning from expert policy
# python run_BC.py expert_data/Humanoid-v2_10_data.pkl Humanoid-v2 --render \ --num_roll_outs 20

# importing important libraries
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
import pickle as pkl
import tensorflow as tf
import tf_util
import gym


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pkl.loads(f.read())
    return data


def train_model(expert_data,env_name):
    # this function used for training the model

    obs_data = expert_data['observations']
    obs_data = np.reshape(obs_data, (obs_data.shape[ 0 ], obs_data.shape[1]))
    act_data = expert_data[ 'actions' ] #same
    act_data = np.reshape(act_data, (act_data.shape[0], act_data.shape[2]))

    # defining model for the behavioral cloning

    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(obs_data.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(act_data.shape[1], activation='linear'))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(obs_data, act_data, epochs=30, batch_size=256, validation_split=0.1, shuffle=True)
    model.save('models/' + env_name + '_cloned_model.h5')


# simulate the data and save it into appropriate file
def simulate_bc(env_name, num_roll_outs, render, max_time_steps):
    with tf.Session():
        tf_util.initialize()
        env = gym.make(env_name)
        max_steps = env.spec.timestep_limit or max_time_steps
        model = load_model('models/' + env_name + '_cloned_model.h5')
        returns = []

        for i in range(num_roll_outs):
            print('iter', i)
            obs = env.reset()
            done = False
            total_reward = 0.
            steps = 0
            while not done:
                obs_bc = np.array(obs)
                obs_bc = np.reshape(obs, (1, obs_bc.shape[0]))
                model_action = model.predict(obs_bc)
                obs, r, done, _ = env.step(model_action)
                total_reward += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(total_reward)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        cache_returns = (returns, np.mean(returns), np.std(returns))
        return cache_returns


def main():
    # importing settings from the command window
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_data_file', type=str)
    parser.add_argument('env_name', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_roll_outs', type=int, default=20, help='Number of expert roll outs')
    parser.add_argument('--max_time_steps', type=int)
    args = parser.parse_args()

    # assigning those settings to variables

    expert_data_file = args.expert_data_file
    env_name = args.env_name
    num_roll_outs = args.num_roll_outs
    render = args.render
    max_time_steps=args.max_time_steps

    expert_data = load_data(expert_data_file)
    train_model(expert_data,env_name)
    return_data = simulate_bc(env_name, num_roll_outs, render, max_time_steps)


if __name__ == "__main__":
    main()
