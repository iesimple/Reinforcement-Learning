# 没有网格地图的限制
# 动作action：up、down
# 状态state："state_{action}_{action}......"
# 限制最大行动次数为5，也就是在状态中有5个action就会结束
# 游戏结束状态只有两个："happy_end" "bad_end" ———————— 这里的两个状态和上面所说的状态其实是对应的，规则：
# state中action为up的次数大于阈值（这里设置为4），就是happy_end，否则就是bad_end
# 主要是为了理解V这个函数，在这里是基于价值最大化来选择行动的情况下价值的最大值


def V(s, gamma=0.99):
    '''
        计算从状态s开始基于价值的bellman方程进行行动选择的价值
    '''
    V = R(s) + gamma * max_V_on_next_state(s)
    return V


def R(s):
    # 定义两种结束状态的奖励值
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0


def max_V_on_next_state(s):
    '''
        计算在当前状态为s时V(s')的最大值
    '''
    # 如果当前状态已经表示游戏结束了，也即没有下个状态了，那显然就是0
    if s in ["happy_end", "bad_end"]:
        return 0
    actions = ["up", "down"]
    values = []  # values记录每一种行动后的最大价值
    # 基于价值最大化选择行动，因此计算每一种动作下的价值，取最大值返回
    for a in actions:
        # 计算转移概率，因为可能会转移到不同的状态，因此这里返回一个字典：key=状态；value=概率
        transition_probs = transit_func(s, a)
        v = 0
        for next_state in transition_probs:
            prob = transition_probs[next_state]  # 转移概率
            v += prob * V(next_state)  # 累加，因为是考虑不同行动，相当于积分掉状态
        values.append(v)
    return max(values)


def transit_func(s, a):
    '''
        计算在当前状态为s，选择的行动为a的情况下所有的下一状态s'以及其对应的概率
    '''
    actions = s.split("_")[1:]  # 这里从1开始和输入形式有关
    LIMIT_GAME_COUNT = 5  # 限制游戏的最大步数
    HAPPY_END_BORDER = 4  # 达到happy_end的阈值，这里所有行动中up的次数
    MOVE_PROB = 0.9  # 转移概率，即实际行动按照计划行动执行的概率

    def next_state(state, action):
        return "_".join([state, action])

    if len(actions) == LIMIT_GAME_COUNT:
        # 行动的次数已经达到最大次数，游戏结束，计算up的次数，返回happy_end或者bad_end
        up_count = sum([1 if a == "up" else 0 for a in actions])
        state = "happy_end" if up_count >= HAPPY_END_BORDER else "bad_end"
        prob = 1.0
        return {state: prob}
    else:
        # 这里行动只有两种，而且不考虑会让游戏结束的状态，因此就是直接返回两种行动之后状态及其对应的概率
        opposite = "up" if a == "down" else "down"
        return {
            next_state(s, a): MOVE_PROB,
            next_state(s, opposite): 1 - MOVE_PROB
        }


# 在不同初始状态下，up多的初始状态最大的价值会更大，注意是最大的价值
if __name__ == "__main__":
    print(V("state"))
    print(V("state_up_up"))
    print(V("state_down_down"))