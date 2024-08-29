from env import env
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
from collections import deque


#using tensorflow, create a model for deep q learning
#this model estimates q values for actions in a state
#uses tensorflow keras to create a sequential model with three layers 
def create_q_model(state_dim, action_dim):
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(64, activation='relu', input_shape=(state_dim,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(action_dim))
    #environment state is passed through the layers to generate approximated q-valeu

    return model

    
    
#create a replay buffer that stores experiences

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    #adf an expereince 
    def add(self, experience):
        self.buffer.append(experience)
    #select a random batch of experiences from the buffer.
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    

def train_step(batch_samples):
    states, actions, rewards, next_states, dones = zip(*batch_samples)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)
    #unpack tuples


    #find the difference between current q-values and target q-values and find the loss
    with tf.GradientTape() as tape:
        current_q = model(states, training=True)
        current_q_values = tf.reduce_sum(current_q * tf.one_hot(actions, action_dim), axis=1)
        next_q_values = tf.reduce_max(target_model(next_states, training=False), axis=1)
        expected_q_values = rewards + discount_factor * next_q_values * (1 - dones)
        loss = loss_fn(expected_q_values, current_q_values)

    #then update the mdel's weights using the gradient descent
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

#uses optimal q values after training
def use_optimised_values():

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        q_values = model.predict(state[np.newaxis])    
        action = np.argmax(q_values[0])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        
        time.sleep(1)


#initialise values for the training
    
#create environment and state and action dimensions
env = env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

#create a model and a target model 
model = create_q_model(state_dim, action_dim)
target_model = create_q_model(state_dim, action_dim)
target_model.set_weights(model.get_weights())

#adam optimiser as it is suitable for a larger dimensional apace
#learning rate to 0.001 and slowly decrease
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.005,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.Huber()

#values for q-learning
replay_buffer = ReplayBuffer()
batch_size = 64
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.1
discount_factor = 0.99



# loop training 1000 times
for episode in range(1000):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
            #select random action if outside of epsilon value
        else:
            action_prob = model.predict(state[np.newaxis])
            action = np.argmax(action_prob)
            #select highest value action greedily

        #step the environment and add performed action to replay buffer
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add((state, action, reward, next_state, done))

        state = next_state
        episode_reward += reward

        if len(replay_buffer.buffer) > batch_size:
            batch_samples = replay_buffer.sample(batch_size)
            train_step(batch_samples)

        if done:
            break
        
    #every 10 episodes set weights of target model to models weight, giving a stable template 
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print("Episode: ", episode, "Reward: ", episode_reward , "Epsilon: " , epsilon)


use_optimised_values()

env.close()


