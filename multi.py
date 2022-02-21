import numpy as np
from itertools import product
from csrl.mdp import GridMDP
import os
import importlib
from copy import deepcopy

import matplotlib.pyplot as plt

from ipywidgets.widgets import IntSlider
from ipywidgets import interact


class MultiControlSynthesis:

    def __init__(self, controls, mdp, starts=[(0, 0), (0, 2)], agents=2, oa = None, sharedoa=False):
        self.nagents = agents  # number of agents
        self.agent_control = controls  # array containing a Control Synthesis object for each agent
        self.get_larger_shape([controls[i].shape for i in range(agents)])
        # self.shape = (self.nagents,) + controls[0].shape
        self.Q = np.zeros(shape=self.shape) # TODO here we take control[0] but that is wrong
        self.starts = starts
        self.mdp = mdp

        self.shared_oa = sharedoa
        self.reward = np.zeros(shape=self.shape[1:-1]+mdp.shape)

        # tODO make a new reward matrix based on both agents
        if sharedoa:
            self.oa = oa
            for i,q,r,c in self.agent_control[0].states():
                for r1, c1 in mdp.states():
                    global_l = (mdp.label[r,c])
                    for label in self.mdp.label[r1, c1]:
                        if not label in global_l:
                            global_l = global_l + (label,)
                    global_l = tuple(sorted(global_l))
                    self.reward[i,q,r,c, r1, c1] =   
                    # if q == oa.shape[1]-1: # trap state
                    #     self.reward[i,q,r,c, r1, c1] = -1

    def get_larger_shape(self, shapes):
        self.shape = (self.nagents,)
        for i in range(len(shapes[0])):
            self.shape += (max([shapes[j][i] for j in range(len(shapes))]),)

    # TODO adapt this for 2 agents make sure to specify starting locations
    def ind_qlearning(self, start=None, T=1000, K=100000):

        for i in range(self.nagents):
            self.Q[i] = self.agent_control[i].q_learning(start=self.starts[i], T=T, K=K)

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
        
        # Q = np.zeros(self.shape[1:-1]+mdp.shape+mdp.shape) 
        # print(Q.shape)
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
                # print("state :",state[0], ' - ', state[0][:2], ' - ',  state[0][2:])
                comb_state= tuple(state[0][:2])
                for i in range(self.nagents):
                    if self.shared_oa:
                        comb_state += tuple(state[i][2:])
                        if i == self.nagents-1:
                            reward = np.ones(shape=(self.nagents,1)) * self.reward[comb_state] 
                    else:
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
                    #print('states: ', states)
                    # print('shape :', self.agent_control[i].shape)
                    next_state[i] = np.array(states[np.random.choice(len(states), p=probs)])

                    # add transition_reward -5 trap, -1 for self loop based on automaton state/transition
                    # try encouraging transitions instead of discouraging self loop?
                    # if (next_state[i][1] !=  state[i][1]) and reward[i] > 0: # self loop
                    #     reward[i] += 0.05

                    # TODO find out how to OR the label using the action. -> understand the action matrix better

                global_labels = ()
                for j in range(self.nagents):
                    for label in self.mdp.label[tuple(next_state[j][2:])]:
                        if not label in global_labels:
                            global_labels = global_labels + (label,)

                global_labels = tuple(sorted(global_labels))
                # labels_temp = [x for x in global_labels if x in self.agent_control[i].oa.all_labels]

                # if len(global_labels) > 0:
                #     print('labels: ', global_labels)
                if self.shared_oa:
                    temp = self.oa.delta[next_state[0][1]][global_labels]

                for i in range(self.nagents):
                    # transition OA states
                    if self.shared_oa:
                        next_state[i][1] = temp
                    else:
                        next_state[i][1] = self.agent_control[i].oa.delta[next_state[i][1]][global_labels]
                    # Q-update
                    self.Q[i][tuple(state[i])][action[i]] += alpha * (reward[i] + gamma[i]*np.max(self.Q[i][tuple(next_state[i])]) - self.Q[i][tuple(state[i])][action[i]])

                    state[i] = deepcopy(next_state[i])
        
        return self.Q

    def combined_qlearning_sharedstate(self, T=None, K=None):
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
        
        Q = np.zeros((self.nagents,)+self.shape[1:-1]+self.mdp.shape+(self.shape[-1],)) 
        print(Q.shape)

        state = np.zeros(shape=(self.nagents*2), dtype=int)
        oa_state = np.zeros(shape=(self.nagents,2), dtype=int)
        next_oa_state = np.zeros(shape=(self.nagents,2), dtype=int)
        action = np.zeros(shape=(self.nagents,1), dtype=int)
        reward = np.zeros(shape=(self.nagents, 1))
        next_state = np.zeros(shape=(self.nagents*2), dtype=int)
        trap = self.agent_control[0].oa.shape[1]-1

        for k in range(K):
            state = (self.starts[0] if self.starts[0] else self.mdp.random_state()) \
                                    +(self.starts[1] if self.starts[1] else self.mdp.random_state())
            for i in range(self.nagents): 
                oa_state[i] = (self.shape[1]-1, self.agent_control[i].oa.q0)

            alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
            epsilon = np.max((1.0*(1 - 1.5*k/K),0.01))
            print(f'------------ episode {k}')

            for t in range(T):
                
                state= tuple(state)

                for i in range(self.nagents):
                    reward[i] = self.reward[tuple(oa_state[i]) + state]

                gamma = [self.agent_control[i].discountB if reward[i] else self.agent_control[i].discount for i in range(self.nagents)]
                
                for i in range(self.nagents):
                    state_i = tuple(oa_state[i])+tuple(state[i*2:(i+1)*2]) 
                    # Follow an epsilon-greedy policy
                    if np.random.rand() < epsilon or np.max(Q[i][tuple(oa_state[i])+tuple(state)])==0:
                        action[i] = np.random.choice(self.agent_control[i].A[state_i])  # Choose among the MDP and epsilon actions
                    else:
                        action[i] = np.argmax(Q[i][tuple(oa_state[i])+ tuple(state)])

                    # Observe the next state
                    states, probs = self.agent_control[i].transition_probs[state_i][action[i]][0]
                    #print('states: ', states)
                    # print('shape :', states, probs)
                    chosen = np.array(states[np.random.choice(len(states), p=probs)])
                    # print(state_i, chosen)
                    next_state[i*2:2+i*2] = chosen[2:]
                    oa_state[i] = chosen[0:2]


                global_labels = ()
                for j in range(self.nagents):
                    for label in self.mdp.label[tuple(next_state[j*2:2+j*2])]:
                        if not label in global_labels:
                            global_labels = global_labels + (label,)

                global_labels = tuple(sorted(global_labels))
                # labels_temp = [x for x in global_labels if x in self.agent_control[i].oa.all_labels]

                # if len(global_labels) > 0:
                #     print(f"trap: {trap} , actions {list(action.flatten())}, oa_state {list(oa_state.flatten())}, state :{state}, nextstate :{tuple(next_state)}",' -- new labels: ', global_labels)
                # else:
                #     print(f"trap: {trap} , actions {list(action.flatten())}, oa_state {list(oa_state.flatten())}, state :{state}, nextstate :{tuple(next_state)}")

                # transition OA states
                flag_trap = np.zeros(self.nagents)
                for i in range(self.nagents):
                    # Q-update
                    # print(Q[i].shape, state, next_state)
                    next_oa_state[i][1] = self.oa.delta[oa_state[i][1]][global_labels]
                    Q[i][tuple(oa_state[i])+state][action[i]] += alpha * (reward[i] \
                                + gamma[i]*np.max(Q[i][tuple(next_oa_state[i])+tuple(next_state)]) - Q[i][tuple(oa_state[i])+state][action[i]])

                    if oa_state[i][1] == trap  and next_oa_state[i][1] == trap : # trap state
                        flag_trap[i] = 1;

                state = deepcopy(tuple(next_state))
                oa_state=deepcopy(next_oa_state)

                if flag_trap.all():
                    break
        
        return Q

    def plot(self, i, value=None, iq=None, **kwargs):
        self.agent_control[i].plot(policy=np.argmax(self.Q[i], axis=4), value=np.max(self.Q[i],axis=4))

    def plot(self, i, policy=None,value=None, iq=None, **kwargs):
        self.agent_control[i].plot(policy=policy, value=value)

    # TODO for multi agent
    def simulate(self, policy, agents, mdp2 =None, start=None, T=None, use_mdp2 =False, qlearning=True, plot=True, animation=None):
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
        T = T if T else 50
        print(T)
        state = []

        for i in range(self.nagents):
            state.append((self.shape[1]-1,agents[i].oa.q0)+self.starts[i])

        episode = [state]
        print('e', episode)

        for t in range(T):
            next_state = []
            for i in range(self.nagents):
                if qlearning:
                    # print(f'agent {i}', policy[i][state[i]])
                    states, probs = agents[i].transition_probs[state[i]][policy[i][state[i]]]
                else:
                    states, probs = agents[i].transition_probs[state[i]][policy[i][state[i]]]
                #print('next:',states[np.random.choice(len(states), p=probs)])
                next_state.append(states[np.random.choice(len(states), p=probs)])

            global_labels = ()
            for j in range(self.nagents):
                for label in self.mdp.label[tuple(next_state[j][2:])]:
                    if not label in global_labels:
                        global_labels = global_labels + (label,)
            global_labels = tuple(sorted(global_labels))

            for i in range(self.nagents):
                temp = list(next_state[i])
                # transition OA states
                temp[1] = self.agent_control[i].oa.delta[next_state[i][1]][global_labels]
                next_state[i] = tuple(temp)

            episode.append(next_state)
            state = next_state

            # if plot:
            #     def plot_agent(t, i=0):
            #         if use_mdp2:
            #             print('e', episode[t][i][0][:2],'policy',self.Q[i][episode[t][i][0][:2]].shape, '\n agent',episode[t][i][0][2:], 'bla')
            #             mdp2.plot(policy=self.Q[i][episode[t][i][0][:2]], agent=episode[t][i][0][2:])
            #         else:
            #             self.agent_control[i].mdp.plot(policy=self.Q[i][episode[t][i][0][:2]], agent=episode[t][i][0][2:])

            #     plot_agent(t=t)
            #print(episode[t][0][:2])
        if animation:
            pad=5
            if not os.path.exists(animation):
                os.makedirs(animation)
            for t in range(T):
                print(t, ': ', episode[t][0][1:], '\t', episode[t][1][1:])
                mdp2.multi_plot(nagents=self.nagents, policy=[policy[0][episode[t][0][:2]], policy[1][episode[t][1][:2]]],
                    agent=[episode[t][0][2:], episode[t][1][2:]],save=animation+os.sep+str(t).zfill(pad)+'_comb.png')
                plt.close()

        return episode

    # TODO for multi agent
    def simulate_shared(self, policy, agents, mdp2 =None, start=None, T=None, use_mdp2 =False, qlearning=True, plot=True, animation=None):
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
        T = T if T else 50
        print(T)
        state = np.zeros(shape=(self.nagents*2), dtype=int)
        oa_state = np.zeros(shape=(self.nagents,2), dtype=int)
        next_oa_state = np.zeros(shape=(self.nagents,2), dtype=int)
        next_state = np.zeros(shape=(self.nagents*2), dtype=int)
        action = np.zeros(shape=(self.nagents,1), dtype=int)

        state = (self.starts[0] if self.starts[0] else self.mdp.random_state()) \
                                +(self.starts[1] if self.starts[1] else self.mdp.random_state())
        for i in range(self.nagents): 
            oa_state[i] = (self.shape[1]-1, self.agent_control[i].oa.q0)

        episode = [tuple(oa_state.flatten())+state]
        print('e', episode)

        for t in range(T):
            
            for i in range(self.nagents):
                state_i = tuple(oa_state[i])+tuple(state[i*2:(i+1)*2]) 

                action[i] = policy[i][tuple(oa_state[i])+ tuple(state)]

                # Observe the next state
                states, probs = self.agent_control[i].transition_probs[state_i][action[i]][0]

                chosen = np.array(states[np.random.choice(len(states), p=probs)])

                next_state[i*2:2+i*2] = chosen[2:]
                oa_state[i] = chosen[0:2]

            global_labels = ()
            for j in range(self.nagents):
                for label in self.mdp.label[tuple(next_state[j*2:2+j*2])]:
                    if not label in global_labels:
                        global_labels = global_labels + (label,)

            global_labels = tuple(sorted(global_labels))

            for i in range(self.nagents):
                next_oa_state[i][1] = self.oa.delta[oa_state[i][1]][global_labels]
            
            episode.append(tuple(oa_state.flatten())+state)

            # if plot:
            #     def plot_agent(t, i=0):
            #         if use_mdp2:
            #             print('e', episode[t][i][0][:2],'policy',self.Q[i][episode[t][i][0][:2]].shape, '\n agent',episode[t][i][0][2:], 'bla')
            #             mdp2.plot(policy=self.Q[i][episode[t][i][0][:2]], agent=episode[t][i][0][2:])
            #         else:
            #             self.agent_control[i].mdp.plot(policy=self.Q[i][episode[t][i][0][:2]], agent=episode[t][i][0][2:])

            #     plot_agent(t=t)
        print(episode[t])
        if animation:
            pad=5
            if not os.path.exists(animation):
                os.makedirs(animation)
            for t in range(T):
                print(t, ': ', ' oa_state:',episode[t][:4], ' state :', episode[t][4:])
                mdp2.multi_plot(nagents=self.nagents, policy=[policy[0][episode[t][:2]+episode[t][6:8]], policy[1][episode[t][:2]+episode[t][4:6]]],
                    agent=[episode[t][4:6], episode[t][6:]],save=animation+os.sep+str(t).zfill(pad)+'_comb.png')
                plt.close()

        return episode


