from environment import Environment
from planner import ValueIterationPlanner, PolicyIterationPlanner


def print_result(result):
    step = 1
    for grid in result:
        print("------------第%s次迭代------------" % (step))
        for row in range(len(grid)):
            for column in range(len(grid[0])):
                print("%6.3f" % (grid[row][column]), end=' ')
            print()
        print("---------------------------------")
        step += 1


def main():
    '''
        测试用主函数
    '''
    print("--------------------输入gird地图--------------------")
    print(
        "输入一个二维数组代表地图，以空格分隔列，回车分隔行\n 0：普通格子\n-1：有危险的格子（游戏结束）\n 1：有奖励的格子（游戏结束）\n 9：被屏蔽的格子"
    )
    row = int(input("首先输入地图的行数："))
    print("按上述规则输入地图：")
    grid = [[int(s) for s in input().split()] for _ in range(row)]
    print("---------------------------------------------------")
    print("----------------------其他输入----------------------")
    plan_type = input("价值迭代 or 策略迭代 [v/p]：")
    move_prob = 0.8
    # move_prob = float(input("移动概率："))
    print("---------------------------------------------------")

    env = Environment(grid, move_prob=move_prob)
    result = []
    if plan_type == "v":
        planner = ValueIterationPlanner(env)
    elif plan_type == "p":
        planner = PolicyIterationPlanner(env)

    result = planner.plan()
    planner.log.append(result)
    print_result(planner.log)


'''
grid
0 0 0 1 
0 9 0 -1
0 0 0 0
'''

if __name__ == "__main__":
    main()