#!/usr/bin/env python3

import random
import gym_gazebo
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')

    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.99
    epsilon_decay = 0.95

    q_table = np.zeros((observation_space.n, action_space.n))

    highest_reward = 0

    LOAD_OLD_MODEL = True
    
    if LOAD_OLD_MODEL:
        try:
            q_table = loadQ("best_q_table.csv")
            epsilon = 0.05  # Override exploration to exploit learned values
            print(f"Loaded existing model. Starting epsilon set to {epsilon}")
        except FileNotFoundError:
            print("No existing Q-table found. Starting from scratch.")

    try:
        state, _ = env.reset()
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False

            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print(f"New highest reward: {total_reward}.")
                saveQ(q_table, "best_q_table.csv")

            env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table):
    alpha = 0.1
    gamma = 0.9

    if next_state == -1:
        best_next_q = 0.0
    else:
        best_next_q = np.max(q_table[next_state])
    
    td_target = reward + gamma * best_next_q
    
    # FIX: Only update the Q-table if the current state is a valid row
    if state != -1:
        td_error = td_target - q_table[state, action]
        q_table[state, action] += alpha * td_error

    return q_table


def choose_action(state, q_table, epsilon, action_space):
    """
    Epsilon-greedy action selection with random tie-breaking.
    """
    if random.random() < epsilon:
        # Exploration: pick a random action from the action space
        return action_space.sample()
    else:
        # Exploitation: pick the action with the highest Q-value
        state_q_values = q_table[state]
        max_q = np.max(state_q_values)
        
        # Identify all actions that share the maximum Q-value
        best_actions = np.where(state_q_values == max_q)[0]
        
        # Break ties randomly among best actions
        return int(np.random.choice(best_actions))


def saveQ(q_table, save_path):
    """
    Saves the Q-table to a CSV file so it can be opened in Excel/text editor.
    """
    np.savetxt(save_path, q_table, delimiter=",")
    print(f"Q-table successfully saved to {save_path}")

def loadQ(load_path):
    """
    Loads a Q-table from a CSV file.
    """
    q_table = np.loadtxt(load_path, delimiter=",")
    print(f"Q-table successfully loaded from {load_path}")
    return q_table

if __name__ == "__main__":
    main()