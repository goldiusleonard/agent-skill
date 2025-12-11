import numpy as np


def _choose_exploration_action(self, state):
    agent_pos = state["agent_pos"]
    if not state.get("has_key"):
        key_pos = state["objects"]["key"]
        if agent_pos == key_pos:
            return "pickup_key"
        if agent_pos[0] < key_pos[0]:
            return "move_right"
        if agent_pos[0] > key_pos[0]:
            return "move_left"
        if agent_pos[1] < key_pos[1]:
            return "move_up"
        return "move_down"
    if state.get("has_key") and not state.get("door_open"):
        door_pos = state["objects"]["door"]
        if agent_pos == door_pos:
            return "open_door"
        if agent_pos[0] < door_pos[0]:
            return "move_right"
        if agent_pos[0] > door_pos[0]:
            return "move_left"
        if agent_pos[1] < door_pos[1]:
            return "move_up"
        return "move_down"
    goal_pos = (4, 4)
    if agent_pos[0] < goal_pos[0]:
        return "move_right"
    if agent_pos[1] < goal_pos[1]:
        return "move_up"
    return np.random.choice(self.primitive_actions)


def run_episode(self, use_skills=True):
    self.env.reset()
    total_reward = 0
    steps = 0
    trajectory = []
    while steps < 50:
        state = self.env.get_state()
        if use_skills and self.skill_library.skills:
            query_emb = self.create_embedding(state, [])
            skills = self.skill_library.retrieve_skills(state, query_emb, top_k=1)
            if skills:
                skill_traj, skill_reward, success = self.execute_skill(skills[0])
                trajectory.extend(skill_traj)
                total_reward += skill_reward
                steps += len(skill_traj)
                if success:
                    return trajectory, total_reward, steps, True
                continue
        action = self._choose_exploration_action(state)
        next_state, reward, done = self.env.step(action)
        trajectory.append((state, action, reward))
        total_reward += reward
        steps += 1
        if done:
            return trajectory, total_reward, steps, True
    return trajectory, total_reward, steps, False


def train(self, episodes=10):
    stats = {"rewards": [], "steps": [], "skills_learned": [], "skill_uses": []}
    for ep in range(episodes):
        trajectory, reward, steps, success = self.run_episode(use_skills=True)
        if success and len(trajectory) >= 3:
            segment = trajectory[-min(5, len(trajectory)) :]
            skill = self.extract_skill(segment)
            if skill:
                self.skill_library.add_skill(skill)
        stats["rewards"].append(reward)
        stats["steps"].append(steps)
        stats["skills_learned"].append(len(self.skill_library.skills))
        stats["skill_uses"].append(self.skill_library.get_stats()["total_uses"])
        print(
            f"Episode {ep + 1}: Reward={reward:.1f}, Steps={steps}, Skills={len(self.skill_library.skills)}, Success={success}"
        )
    return stats
