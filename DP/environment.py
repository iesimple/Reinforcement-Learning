from enum import Enum
from re import S
import numpy as np
import random


class State():
    '''
        状态类，这里是在网格地图中的每个位置作为一个状态
    '''
    def __init__(self, row=-1, column=-1) -> None:
        '''
            构造函数，以网格地图的二维坐标作为状态标识

            row - 行

            column - 列
        '''
        self.row = row
        self.column = column

    def __repr__(self) -> str:
        '''
            打印当前状态
        '''
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        '''
            克隆当前状态，返回一个复制
        '''
        return State(self.row, self.column)

    def __hash__(self) -> int:
        '''
            返回对象的hash值，不知道有什么用
        '''
        return hash((self.row, self.column))

    def __eq__(self, other) -> bool:
        '''
            设置对象的等于判定逻辑
        '''
        return self.row == other.row and self.column == other.column


class Action(Enum):
    '''
        动作类，四个方向
    '''
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


# @property
# 1. 修饰方法，使方法可以像类的属性一样直接调用，而不用加括号；加了反而会报错
# 2. 与所定义的属性配合使用，防止修改，因为python类里没有类似c++的私有成员变量


class Environment():
    def __init__(self, grid, move_prob=0.8) -> None:
        # grid是一个二维数组，它的值可以看作属性
        # 0：普通格子
        # -1：有危险的格子（游戏结束）
        # 1：有奖励的格子（游戏结束）
        # 9：被屏蔽的格子
        self.grid = grid
        self.agent_state = State()
        # 默认奖励设置为负数，意味着agent必须快速到达终点
        self.default_reward = -0.04
        # 以多大的概率向所选择的方向移动
        self.move_prob = move_prob

        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # state中不包含被屏蔽的格子
                if self.grid[row][column] != 9:
                    states.append(State(row, column))

        return states

    def transit_func(self, state, action):
        transition_probs = {}
        if not self.can_action_at(state):
            # 到达游戏结束的格子
            return transition_probs
        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob
        return transition_probs

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # 执行移动
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # 检查状态是否在grid外
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        '''
            奖励函数，根据当前状态返回奖励值以及游戏是否结束
        '''
        reward = self.default_reward
        done = False

        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            reward = 1
            done = True
        elif attribute == -1:
            reward = -1
            done = True

        return reward, done

    def reset(self):
        '''
            将agent放到左下角，重新开始
        '''
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)

        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, None

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)
        return next_state, reward, done

