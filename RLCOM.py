"""
RLCOM
"""
# -*- coding: <utf-8> -*-

import numpy as np
import heapq

"""
N: number of base stations
C: capacity of each stations
I: total numbers of corporate users

state:
    user distribution on N base stations
    a 2N-element array (i_1, ..., i_n, o_1, ..., o_n)
    ip and op represents corporate users and guest users respectively, where
        p is in {1, 2, ..., n}
action:
    redistribution of corporate users
    a N-element array (i`_1, ..., i`_n)
"""

N = 4
C = 4
I = 8
ACTIONS = "01234"

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.15
OMEGA = 0.5
EPISODE_NUM = 6000000
STEP_NUM = 30

def pop(s):
    if len(s) > 0:
        ch = s[-1]
        s = s[:-1]
        return s, ch
    else:
        return ""

def push(s, ch):
    """
    s and ch are strings
    push function concatenate ch behind s and returns the new string
    """
    return s + ch

def list2str(list):
    """
    list2str: given a list of int numbers, turn it into string
    Parameter type:
        list: list of int numbers
    """
    result = ""
    for num in list:
        result += str(num)
    return result

def cal_num_states(base, capacity, coruser):
    """
    cal_num_states: returns total number of possible states
    Parameter list:
        base    :   num of base stations
        capacity:   capacity of each base station
        coruser :   num of corporate users
    """
    if base == 1:
        return max(0, capacity - coruser + 1)
    n_states = 0
    for i in range(min(capacity, coruser) + 1):
        n_states += (capacity - i + 1) * cal_num_states(base - 1, capacity, coruser - i)
    return n_states

def is_legal_state(state):
    """
    is_legal_state: check if a given state is legal, return True if yes, False otherwise
    Parameter list:
        state: a state in form of list
    """
    is_legal = (len(state) == 2*N) and (sum(state[:N]) == I)
    for i in range(N):
        is_legal = is_legal and ((state[i] + state[i+N]) <= C)
    return is_legal

def get_actions(state):
    """
    get_actions: given a legal state, return a list of (-state_action_value, action_str)
        the state_action_value is initialized as 5, encouraging exploring
    Parameter list:
        state: a legal state in form of list
    """
    action = [0]
    actions = []
    while True:
        curr_act = action[-1]
        curr_index = len(action) - 1
        if curr_act + state[curr_index + N] > C:
            action.pop()
            if len(action) > 0:
                action[-1] += 1
            else:
                break
        elif len(action) < N:
            action.append(0)
        else:
            if sum(action) == I:
                heapq.heappush(actions, (-5 + np.random.normal(), list2str(action)))
            action[-1] += 1
    return actions

def init_table():
    """
    init_table: given N, C, I, initiate the table in Q-Learning
    the table is a dict, whose keys are strings like "22221221" representing
        current state (i_1, ..., i_4, o_N, ...., o_N), and the corresponding
        value is a list of tuples (-Q(s,a), a), in which a is a string like
        "3023" representing a redistribution of corporate users
    """
    Q = {}
    state = [0]
    while True:
        curr_act = state[-1]
        if curr_act > C:
            state.pop()
            if len(state) > 0:
                state[-1] += 1
            else:
                break
        elif len(state) == N:
            if sum(state[:N]) < I:
                state[-1] += 1
            elif sum(state[:N]) > I:
                state.pop()
                state[-1] += 1
            else:
                state.append(0)
        elif len(state) < 2*N:
            state.append(0)
        else:
            if is_legal_state(state):
                state_str = list2str(state)
                Q[state_str] = get_actions(state)
            state[-1] += 1
    return Q

def get_reward(s_t, a_t, s_t1):
    """
    get_reward: get reward given s_t, a_t, s_t1 and return it
    Parameter list:
        s_t  : state at time slot t, string format
        a_t  : action at time slot t, string format
        s_t1 : state at time slot t+1, string format
    """
    reward = 0.

    for bs in range(N):
        i_n = int(s_t1[bs])
        o_n = int(s_t1[bs + N])
        reward = (i_n + OMEGA * o_n) * np.log(1. / (i_n + o_n))

    return reward

def redistribute_gu_uniform(s_mid):
    """
    redistribute_gu: redistribute guest users after the agent takes an action
        --- redistribute the corporate users. This function uniformly selects
        an interger between [0, C - corp_num] for each base station
    Parameter list:
        s_mid: a middle state after the agent takes the action and before the
            dynamics of guest users makes effects
    """
    guest_users = []
    for n in range(N):
        guest_users.append(
            np.rand.randint(0, N - int(s_mid[n]) + 1)
            )
    return s_mid[:N] + list2str(guest_users)

def take_action(s_t, a_t):
    """
    take_action: given state and action at time slot t, take the action and
    transit to a new state s_t1 by some rule of dynamics of guest users,
    then calculate the reward r_t1 = r(s_t, a_t, s_t1) and return r_t1, s_t1
    Parameter list:
        s_t: state at time slot t, string format
        a_t: action at time slot t, string format
    """
    s_t1 = redistribute_gu(a_t + s_t[N:])
    r_t1 = get_reward(s_t, a_t, s_t1)
    return r_t1, s_t1

def train_Q(Q):
    """
    train_Q: train the Q function in EPISODE_NUM episodes
    Parameter list:
        Q: the Q function
    """
    for episode in range(EPISODE_NUM):
        # initiate a state and get its action sets
        s_t, A_t = Q.items()[np.random.randint(0, len(Q))]

        for t in range(STEP_NUM):
            # epsilon-greedily choose an action a_t from A_{s_t}
            if np.random.rand() < EPSILON:
                action_index = np.random.randint(len(A_t))
            else:
                # greedily choose the best action in current situation
                action_index = 0
            minus_q_s_a, a_t = A_t[action_index]
            q_s_a =  -minus_q_s_a
            
            # take action a_t and observe r_{t+1} and s_{t+1}
            r_t1, s_t1 = take_action(s_t, a_t)
            new_q_s_a = q_s_a + ALPHA * (r_t1 - GAMMA * Q[s_t1][0][0] - q_s_a)
            Q[s_t][action_index] = (-new_q_s_a, a_t)
            heapq.heapify(Q[s_t])

            # next state
            s_t = s_t1
            A_t = Q[s_t1]

if __name__ == '__main__':
    Q = init_table()
    num = 0
    for key, value in Q.items():
        num += len(value)
    print(num)
