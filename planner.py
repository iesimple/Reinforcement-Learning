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
        self.log = []

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
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]

        return grid


class ValueIterationPlanner(Planner):
    def __init__(self, env) -> None:
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        self.initialize()
        actions = self.env.actions
        V = {}
        for s in self.env.states:
            V[s] = 0

        while True:
            delta = 0
            self.log.append(self.dict_to_grid)
            for s in V:
                if not self.env.can_action_at(s):
                    continue
                expected_rewards = []
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))
                V[s] = max_reward

            if delta < threshold:
                break

        V_grid = self.dict_to_grid
        return V_grid
