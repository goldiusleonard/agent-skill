class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.objects = {"key": [2, 2], "door": [3, 3], "box": [1, 3]}
        self.inventory = []
        self.door_open = False
        return self.get_state()

    def get_state(self):
        return {
            "agent_pos": tuple(self.agent_pos),
            "has_key": "key" in self.inventory,
            "door_open": self.door_open,
            "at_goal": self.agent_pos == self.goal_pos,
            "objects": {k: tuple(v) for k, v in self.objects.items()},
        }

    def step(self, action):
        reward = -0.1
        if action == "move_up":
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.size - 1)
        elif action == "move_down":
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == "move_left":
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == "move_right":
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.size - 1)
        elif action == "pickup_key":
            if self.agent_pos == self.objects["key"] and "key" not in self.inventory:
                self.inventory.append("key")
                reward = 1.0
        elif action == "open_door":
            if self.agent_pos == self.objects["door"] and "key" in self.inventory:
                self.door_open = True
                reward = 2.0
        done = self.agent_pos == self.goal_pos and self.door_open
        if done:
            reward = 10.0
        return self.get_state(), reward, done
