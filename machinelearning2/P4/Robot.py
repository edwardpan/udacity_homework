import random
import math

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha0 = alpha
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        self.t = self.t + 1
        if self.testing:
            # 1. No random choice when testing
            self.epsilon = 1
        else:
            # 2. Update parameters when learning
            # self.epsilon = math.pow(self.epsilon0, 1 + 0.002 * self.t)
            # self.epsilon = self.epsilon0 * math.pow(0.99, self.t)
            self.epsilon = self.epsilon0 * min(math.cos(self.t/200), 0)
        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """
        # 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        if not self.Qtable.get(state):
            self.Qtable[state] = {'u':0, 'd':0, 'l':0, 'r':0}

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():
            # 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            return random.random() < self.epsilon

        if self.learning:
            if is_random_exploration():
                # 6. Return random choose aciton
                return random.choice(self.valid_actions)
            else:
                # 7. Return action with highest q value
                state_q = self.Qtable[self.state]
                return max(state_q, key=state_q.get)
        elif self.testing:
            # 7. choose action with highest q value
            state_q = self.Qtable[self.state]
            return max(state_q, key=state_q.get)
        else:
            # 6. Return random choose aciton
            actionIndex = int(random.uniform(0, 4))
            return random.choice(self.valid_actions)

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            # 8. When learning, update the q table according
            # to the given rules
            # 目的地的最大Q值
            next_state_max_q = max(self.Qtable[next_state].values())
            # 计算新的Q值
            old_q = self.Qtable[self.state][action]
            # 使用当前奖励与平常奖励做偏差比较决定学习的权重
            abs_vs = [abs(v) for v in self.maze.reward.values()]
            max_r = max(abs_vs)
            min_r = min(abs_vs)
            alpha = (abs(r) - min_r) / (max_r - min_r) * self.alpha + self.alpha
            new_q = (1 - alpha) * old_q + alpha * (r + self.gamma * next_state_max_q)
            self.Qtable[self.state][action] = new_q


    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        self.state = next_state
        self.action = action
        return action, reward
