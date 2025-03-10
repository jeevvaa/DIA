import numpy as np
import tensorflow as tf
import flappy_bird_gym
from collections import defaultdict
import random
from datetime import datetime
import pickle

# Initialize the environment
env = flappy_bird_gym.make("FlappyBird-v0")

# TensorBoard setup
log_dir = "logs/flappy_bird/" + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)

# Parameters for Q-Learning
q_table = defaultdict(lambda: np.zeros(env.action_space.n))
alpha = 0.2  # Lower starting learning rate
alpha_min = 0.01
alpha_decay = 0.999  # Slower decay

epsilon = 0.8  # Slightly lower initial exploration
epsilon_min = 0.01
epsilon_decay = 0.999  # Slower decay

gamma = 0.95  # Slightly lower discount factor to prioritize immediate rewards

episodes = 10000

def choose_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Q-learning training loop
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        action = choose_action(str(state))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        
        # Q-learning update
        best_next_action = np.argmax(q_table[str(next_state)])
        td_target = reward + gamma * q_table[str(next_state)][best_next_action]
        td_error = td_target - q_table[str(state)][action]
        q_table[str(state)][action] += alpha * td_error
        
        state = next_state
        steps += 1
        
        if done:
            break
    
    # Update epsilon and alpha with decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    alpha = max(alpha_min, alpha * alpha_decay)

    # Log to TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar('Episode reward', total_reward, step=episode)
        tf.summary.scalar('Epsilon', epsilon, step=episode)
        tf.summary.scalar('Alpha', alpha, step=episode)
    summary_writer.flush()

    # Print info
    if episode % 100 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon}, Alpha: {alpha}")

env.close()

q_table_filename = 'q_table_trained_4.pkl'
with open(q_table_filename, 'wb') as f:
    pickle.dump(dict(q_table), f)  # Convert defaultdict to dict for pickling

print(f"Q-table saved to {q_table_filename}")
