import gym
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras.optimizers import SGD, Adam
from keras import regularizers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', help='model to load', type=str)
parser.add_argument('-oa', help='activation', default='sigmoid', type=str)
parser.add_argument('-l', help='layers', action='append')
args = parser.parse_args()

layers = [int(i) for i in args.l[0].split(',')]

def create_first_gen(num_individuals=10,output_activation=args.oa):
    generation = []
    for i in range(num_individuals):
        model = Sequential()
        model.add(Dense(4,input_shape=(4,),activation='relu'))
        model.add(Dense(2,activation='relu'))
        model.add(Dense(1,activation=output_activation))
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=.1),metrics=['accuracy'])
        generation.append(model)
    return generation

def s(obs,num=4):
    return np.reshape(obs, [1, num])

def watch_model(individual):
    env = gym.make('CartPole-v0')
    highscore = 0
    for i_episode in range(1):
        print(i_episode)
        obs = env.reset()
        points = 0
        for t in range(500):
            env.render()
            action = individual.predict_classes(s(obs))
            obs, reward, done, _ = env.step(action[0][0])
            points += reward
            if done:
                env.reset()
                if points > highscore:
                    highscore = points
                    print("Episode finished after {} timesteps".format(t+1))
                    break
    print('highscore:',highscore)
    env.close()

def create_model(layers=[4,4,2,1]):
    model = Sequential()
    model.add(Dense(layers[1],input_shape=(layers[0],),activation='relu'))
    for neurons_in_layer in layers[2:-1]:
        model.add(Dense(neurons_in_layer,activation='relu'))
    model.add(Dense(layers[-1],activation='sigmoid'))
    return model

# create/load model
model = create_model(layers=layers)
model.load_weights(args.m)
watch_model(model)
