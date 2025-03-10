import pygame
import flappy_bird_gym
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import time
import csv
import os
import os
from datetime import datetime

# Setup TensorBoard Logging
log_dir = "logs/flappy_bird/" + datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)

# Setup Pygame for action inputs
pygame.init()
def human_play_collect_data(file_path, summary_writer, episode):
    env = flappy_bird_gym.make("FlappyBird-v0")
    clock = pygame.time.Clock()
    actions, states = [], []
    total_reward = 0

    obs = env.reset()
    done = False
    while not done:
        env.render()

        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                    action = 1

        states.append(obs.tolist())  # Convert numpy array to list
        actions.append(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        clock.tick(30)  # Set the FPS to 30

    env.close()

    # Logging total reward to TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar("Total Reward", total_reward, step=episode)
        tf.summary.scalar("Actions Taken", len(actions), step=episode)

    # Append data to CSV
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for state, action in zip(states, actions):
            writer.writerow(state + [action])

    return states, actions

def train_model(states, actions, summary_writer):
    X = np.array(states)
    y = np.array(actions)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model definition
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model with TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
    model.save('flappy_bird_imitation_model.h5')
    return model


def ai_play(model):
    env = flappy_bird_gym.make("FlappyBird-v0")
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs = np.array([obs])
        action = np.argmax(model.predict(obs)[0])
        obs, reward, done, info = env.step(action)
        time.sleep(1/30)  # Control the frame rate for smoother visualization
    
    env.close()


def main():
    file_path = "flappy_bird_data.csv"

    # Check if the file exists and create it with headers if it doesn't
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            headers = [f'state_{i}' for i in range(2)] + ['action']
            writer.writerow(headers)

    states, actions = [], []
    for episode in range(5):
        states_play, actions_play = human_play_collect_data(file_path, summary_writer, episode)
        states.extend(states_play)
        actions.extend(actions_play)

    model = train_model(states, actions, summary_writer)
    ai_play(model)

if __name__ == "__main__":
    main()
