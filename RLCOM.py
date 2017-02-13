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
    index = 0
    actions = []
    while True:
        curr_act = action[-1]
        if curr_act + state[index] > C:
            action.pop()
            if len(action) > 0:
                action[-1] += 1
            else:
                break
        elif len(action) < N:
            action.append(0)
        else:
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

def reward(state, action):
    """
    Reward function: Returns a reward given state s and action a
    Parameter list:
        state  : the d
        action : d
    """
    return state * action

if __name__ == '__main__':
    Q = init_table()
    print(len(Q))

