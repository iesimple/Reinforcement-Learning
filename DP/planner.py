# 用于计划的类，作为价值迭代和策略迭代的基类
# yield
# 以yield标识返回的函数在python中的含义是生成器，不是一个函数了！
# 不能直接调用，而是像对象一样，首先生成一个生成器对象，再使用它的内置函数或者其他方式
# 相比于像list之类的容器，生成器边迭代边生成
# 实际执行的时候，yield类似于return，执行到就返回一个值同时不再往下执行
# 区别在于在使用生成器的内置函数next来生成下一个值的时候，会从上一次yield的地方继续执行


class Planner():
    def __init__(self, env) -> None:
        self.env = env
        self.log = [] # 

    def initialize(self):
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        raise Exception("Planner have to implements plan method.")

    def transitions_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            reward, _ = self.env.reward_func(next_state)
            yield prob, next_state, reward

    def dict_to_grid(self, state_reward_dict):
        '''
            把输入的"状态-价值" 字典转变成数组的形式存储，数组的每个单元格就是对应地图上的状态，值是价值
            输入
                state_reward_dict
                    字典 - key=状态 value=价值
            输出
                grid
                    数组 - 大小与地图一致，值是价值
        '''
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]

        return grid
# #
# 价值迭代
# 1. 初始化所有状态的价值，记为V
# 2. 遍历所有状态，对于某个状态s，利用价值迭代函数直接计算出其价值
# ···注意这里计算所需用到的下个状态的价值要么已经初始化，要么已经计算过，总之是已知的
# 3. 记录所有状态中每次迭代后价值的最大改变量，小于阈值即迭代结束
# #
class ValueIterationPlanner(Planner):
    def __init__(self, env) -> None:
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        actions = self.env.actions
        V = {} # "状态-价值" 字典
        # 初始化各个状态的价值
        for s in self.env.states:
            V[s] = 0

        while True:
            delta = 0
            # 这里存储价值是为了在网页端显示的时候可以看到数据变化
            # 在线测试的时候在价值开始变化的时候其实已经计算完成了，变化的数据是在计算中记录的
            self.log.append(self.dict_to_grid(V))
            # 遍历所有状态
            for s in V:
                # 终止状态
                if not self.env.can_action_at(s):
                    continue
                # 期望奖励
                expected_rewards = []
                # 遍历当前状态下所有可行动作，选择一个最大价值更新V
                for a in actions:
                    r = 0   # 用于记录在状态为s, 选择动作为a时的价值，注意这里已经不是期望奖励了
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                max_reward = max(expected_rewards)  # 价值最大化
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold:
                break

        V_grid = self.dict_to_grid(V)
        return V_grid


# 策略迭代
class PolicyIterationPlanner(Planner):

    def __init__(self, env):
        super().__init__(env)
        self.policy = {}

    def initialize(self):
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                # 初始化策略
                # 一开始时各种行动的概率都是一样的
                self.policy[s][a] = 1 / len(actions)

    def estimate_by_policy(self, gamma, threshold):
        '''
            计算当前策略下每个状态的期望奖励
            输入
                gamma
        '''
        V = {}
        for s in self.env.states:
            # 初始化各种状态的期望奖励
            V[s] = 0

        while True:
            delta = 0
            for s in V:
                expected_rewards = []
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += action_prob * prob * \
                             (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value
            if delta < threshold:
                break

        return V

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)

        while True:
            update_stable = True
            # 在当前的策略下估计期望奖励
            V = self.estimate_by_policy(gamma, threshold)
            self.log.append(self.dict_to_grid(V))

            for s in states:
                # 在当前的策略下得到行动
                policy_action = take_max_action(self.policy[s])

                # 与其他行动比较
                action_rewards = {}
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    action_rewards[a] = r
                best_action = take_max_action(action_rewards)
                if policy_action != best_action:
                    update_stable = False

                # 更新策略（设置 best_action prob=1, otherwise=0 (贪婪)）
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            if update_stable:
                # 如果策略没有更新，则停止迭代
                break

        # 将字典转换为二维数组
        V_grid = self.dict_to_grid(V)
        return V_grid
