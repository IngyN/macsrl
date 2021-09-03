import numpy as np
from itertools import product
from csrl.mdp import GridMDP
import os
import importlib

import matplotlib.pyplot as plt

from ipywidgets.widgets import IntSlider
from ipywidgets import interact


class MultiControlSynthesis:

    def __init__(self, controls, mdp, starts=[(0, 0), (0, 2)], agents=2):
        self.nagents = agents  # number of agents
        self.agent_control = controls  # array containing a Control Synthesis object for each agent
        self.shape = (self.nagents,) + controls[0].shape
        self.Q = np.zeros(shape=self.shape)
        self.starts = starts
        self.mdp = mdp

    # TODO adapt this for 2 agents make sure to specify starting locations
    def ind_qlearning(self, start=None, T=None, K=None):

        for i in range(self.nagents):
            self.Q[i] = self.agent_control[i].q_learning(start=self.starts[i], T=1000, K=100000)

    def combined_qlearning(self, T=None, K=None):
        """Performs the Q-learning algorithm for 2 agents while triggering events and returns the action values.
        
        Parameters
        ----------
        start : int
            The start state of the MDP.
            
        T : int
            The episode length.
        
        K : int 
            The number of episodes.
            
        Returns
        -------
        Q: array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions) 
            The action values learned.
        """
        
        T = T if T else np.prod(self.shape[1:-1])
        K = K if K else 100000
        
        #Q = np.zeros(self.shape) 
        state = np.zeros(shape=(self.nagents,4), dtype=int)
        action = np.zeros(shape=(self.nagents,1), dtype=int)
        reward = np.zeros(shape=(self.nagents, 1))
        next_state = np.zeros(shape=(self.nagents,4), dtype=int)

        for k in range(K):
            state[0] = (self.shape[1]-1, self.agent_control[0].oa.q0)+(self.starts[0] if self.starts[0] else self.mdp.random_state())
            state[1] = (self.shape[1]-1, self.agent_control[1].oa.q0)+(self.starts[1] if self.starts[1] else self.mdp.random_state())

            alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
            epsilon = np.max((1.0*(1 - 1.5*k/K),0.01))

            for t in range(T):
                print(state)
                for i in range(self.nagents):
                    reward[i] = self.agent_control[i].reward[tuple(state[i])]

                gamma = [self.agent_control[i].discountB if reward[i] else self.agent_control[i].discount for i in range(self.nagents)]
                
                for i in range(self.nagents):
                    # Follow an epsilon-greedy policy
                    if np.random.rand() < epsilon or np.max(self.Q[i][tuple(state[i])])==0:
                        action[i] = np.random.choice(self.agent_control[i].A[tuple(state[i])])  # Choose among the MDP and epsilon actions
                    else:
                        action[i] = np.argmax(self.Q[i][tuple(state[i])])

                    # Observe the next state
                    states, probs = self.agent_control[i].transition_probs[tuple(state[i])][action[i]][0]
                    next_state[i] = np.array(states[np.random.choice(len(states), p=probs)])

                for i in range(self.nagents):

                    # TODO find out how to OR the label using the action. -> understand the action matrix better

                    global_labels = ()
                    for j in range(self.nagents):
                        if not self.mdp.label[tuple(next_state[j][2:])] in global_labels:
                            global_labels = global_labels + self.mdp.label[tuple(next_state[j][2:])]
                    global_labels = tuple(sorted(global_labels))
                    # transition OA states
                    next_state[i][1] = self.agent_control[i].oa.delta[next_state[i][1]][global_labels]
                    
                    # Q-update
                    self.Q[i][tuple(state[i])][action[i]] += alpha * (reward[i] + gamma[i]*np.max(self.Q[i][tuple(next_state[i])]) - self.Q[i][tuple(state[i])][action[i]])

                    state[i] = next_state[i]
        
        return self.Q

    def plot(self, i, value=None, iq=None, **kwargs):
        self.agent_control[i].plot(iq=(0,2),policy=np.argmax(self.Q[i], axis=4), value=np.max(self.Q[i],axis=4))

    # TODO for multi agent
    def simulate(self, policy, start=None, T=None, qlearning=True, plot=True, animation=None):
        """Simulates the environment for multiple agents and returns a trajectory obtained under the given policy.

            Parameters
            ----------
            policy : array, size=(nagents, n_pairs,n_qs,n_rows,n_cols)
                The policy for each agent.

            start : int
                The start state of the MDP.

            T : int
                The episode length.

            plot : bool
                Plots the simulation if it is True.

            qlearning : bool
                If given a Q table instead of policies.

            Returns
            -------
            episode: list
                A sequence of states

            """
        T = T if T else np.prod(self.shape[:-1])

        state = []
        for i in range(self.nagents):
            state.append((self.shape[1] - 1, self.agent_control[i].oa.q0)
                         + (start if start else self.agent_control[i].random_state()))
        episode = [state]

        for t in range(T):
            for i in range(self.nagents):
                if qlearning:
                    states, probs = self.agent_control[i].transition_probs[state[i]][np.argmax(policy[state])]
                else:
                    states, probs = self.agent_control[i].transition_probs[state[i]][policy[state]]
                state[i] = states[np.random.choice(len(states), p=probs)]
            episode.append(state)

        if plot:
            def plot_agent(i,t):
                self.agent_control[i].mdp.plot(policy=policy[episode[t][:2]], agent=episode[t][2:])

            for i in range(self.nagents):
                t = IntSlider(value=0, min=0, max=T - 1)
                interact(plot_agent, i=i, t=t)

        return episode


