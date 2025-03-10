# Flappy Bird Gym Agent

This project includes a machine learning agent for the Flappy Bird game, implemented using the Flappy Bird Gym environment.

## Installation



Navigate to the `flappy-bird-gym` directory and install the environment:

```bash
cd flappy-bird-gym
pip install -e .
```

Then, install the remaining project dependencies:

```bash
cd ..
pip install -r requirements.txt
```

## Running the Agent

There are several Python scripts that you can run to train and visualize the agent:

- `Qlearning.py`: Train the agent using Q-learning.
- `imitation_learning.py`: Train the agent using imitation learning.
- `run_simulation_Q_learning.py`: Run a set of simulations using the Q-learning agent.

To train the agent and generate simulation videos, run the corresponding script:

```bash
python Qlearning.py
```

## Visualizing Results with TensorBoard

To visualize the training results, you can use TensorBoard. Launch TensorBoard with the following command:

```bash
tensorboard --logdir=logs
```

Navigate to the URL provided by TensorBoard in your web browser to view the training metrics and results.

## Exporting Simulation Videos

The simulations generate video files which can be found in the project directory:

- `flappy_bird_simulation_1.mp4`
- `flappy_bird_simulation_2.mp4`
- ...
- `flappy_bird_simulation_5.mp4`

You can view these videos with any standard media player to see how the trained agent performs.
```
