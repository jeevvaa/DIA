import flappy_bird_gym
import numpy as np
import cv2
import pickle
import time
from collections import defaultdict

# Initialize the environment
env = flappy_bird_gym.make('FlappyBird-rgb-v0')

# Load the Q-table
def load_q_table(filename):
    with open(filename, 'rb') as f:
        return defaultdict(lambda: np.zeros(env.action_space.n), pickle.load(f))

q_table = load_q_table('q_table_trained_3.pkl')

# Choose action function with low epsilon for minimal exploration
def choose_action(state, epsilon=0.001):
    return np.argmax(q_table[state])  # Exploit

# Function to perform simulations and record them using OpenCV
def simulate_and_record_cv(env, num_simulations):
    for i in range(num_simulations):
        # Define the codec and create VideoWriter object for each simulation
        video_filename = f'flappy_bird_simulation_{i+1}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, 30.0, (288, 512))  # Adjust resolution if needed
        
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = choose_action(str(obs), epsilon=0.0)
            obs, reward, done, info = env.step(action)
            if obs is not None:
                frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            time.sleep(0.033)  # Sleep to maintain ~30 FPS

        print(f"Simulation {i+1} complete")
        out.release()  # Release the video writer after each simulation

    env.close()
    print("All simulations recorded and saved.")

# Run the simulation and recording function
simulate_and_record_cv(env, 5)
