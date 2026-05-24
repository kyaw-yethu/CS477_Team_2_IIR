#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import time

import gym
import matplotlib.pyplot as plt
import gym_gazebo2
import numpy as np
import random

import os
os.system("kill $(ps aux | grep 'gz' | awk '{print $2}')")
os.system("kill $(ps aux | grep 'ros' | awk '{print $2}')")

MAX_ITERATIONS = 10**3 # 50
NUM_EPISODES = 10000

def main():

    env = gym.make('CartPole-v0')
    env.theta_threshold_radians = 30./180.*np.pi

    for episode in range(NUM_EPISODES):
        observation = env.reset()
        observation = env.reset()

        # log_i, log_x, log_xdot, log_theta, log_thetadot, log_F = [], [], [], [], [], []
        
        pos_error = 0
        vel_error = 0
        integral  = 0
        
        iteration = 0

        while True:
            iteration += 1

            # -------------------------------------------------
            # ADD YOUR CODE
            #--------------------------------------------------
            # Design your (serial or parallel) PID controller
            Kp_theta = 145.0

            x         = observation[0]
            x_dot     = observation[1]
            theta     = observation[2]
            theta_dot = observation[3]

            F = Kp_theta * theta

            observation, reward, done, info = env.step(F)

            # Log
            # log_i.append(iteration)
            # log_x.append(x)
            # log_xdot.append(x_dot)
            # log_theta.append(theta)
            # log_thetadot.append(theta_dot)
            # log_F.append(F)
            # -------------------------------------------------
            if not done and iteration>MAX_ITERATIONS:                
                break            
            if done:
                break

        # Plot this episode
        # print(f"Episode {episode} finished after {iteration} iterations.")
        # i = np.array(log_i)
        # fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        # axes[0].plot(i, log_theta, label=r'$\theta$ (rad)')
        # axes[0].plot(i, log_thetadot, label=r'$\dot\theta$ (rad/s)', alpha=0.7)
        # axes[0].axhline(env.theta_threshold_radians, color='r', linestyle='--', alpha=0.3)
        # axes[0].axhline(-env.theta_threshold_radians, color='r', linestyle='--', alpha=0.3)
        # axes[0].set_ylabel('pole state')
        # axes[0].legend(); axes[0].grid(True)

        # axes[1].plot(i, log_F, label='F (N)', color='k')
        # axes[1].set_ylabel('control force')
        # axes[1].set_xlabel('iteration')
        # axes[1].legend(); axes[1].grid(True)

        # fig.suptitle(f'P controller, episode {episode}, Kp={Kp_theta}')
        # fig.tight_layout()
        # fig.savefig(f'p_episode_{episode}.png', dpi=120)
        # plt.close(fig)

    env.close()
            

if __name__ == "__main__":
    main()
