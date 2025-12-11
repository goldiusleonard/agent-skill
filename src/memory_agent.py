import numpy as np

from .skill import SkillLibrary, Skill


class ProceduralMemoryAgent:
    def __init__(self, env, embedding_dim=8):
        self.env = env
        self.skill_library = SkillLibrary(embedding_dim)
        self.embedding_dim = embedding_dim
        self.episode_history = []
        self.primitive_actions = [
            "move_up",
            "move_down",
            "move_left",
            "move_right",
            "pickup_key",
            "open_door",
        ]

    def create_embedding(self, state, action_seq):
        state_vec = np.zeros(self.embedding_dim)
        state_vec[0] = hash(str(state["agent_pos"])) % 1000 / 1000
        state_vec[1] = 1.0 if state.get("has_key") else 0.0
        state_vec[2] = 1.0 if state.get("door_open") else 0.0
        for i, action in enumerate(action_seq[: self.embedding_dim - 3]):
            state_vec[3 + i] = hash(action) % 1000 / 1000
        return state_vec / (np.linalg.norm(state_vec) + 1e-8)

    def extract_skill(self, trajectory):
        if len(trajectory) < 2:
            return None
        start_state = trajectory[0][0]
        actions = [a for _, a, _ in trajectory]
        preconditions = {
            "has_key": start_state.get("has_key", False),
            "door_open": start_state.get("door_open", False),
        }
        end_state = self.env.get_state()
        if end_state.get("has_key") and not start_state.get("has_key"):
            name = "acquire_key"
        elif end_state.get("door_open") and not start_state.get("door_open"):
            name = "open_door_sequence"
        else:
            name = f"navigate_{len(actions)}_steps"
        embedding = self.create_embedding(start_state, actions)
        return Skill(name, preconditions, actions, embedding, success_count=1)

    def execute_skill(self, skill):
        skill.times_used += 1
        trajectory = []
        total_reward = 0
        for action in skill.action_sequence:
            state = self.env.get_state()
            next_state, reward, done = self.env.step(action)
            trajectory.append((state, action, reward))
            total_reward += reward
            if done:
                skill.success_count += 1
                return trajectory, total_reward, True
        return trajectory, total_reward, False

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

    def explore(self, max_steps=20):
        trajectory = []
        state = self.env.get_state()
        for _ in range(max_steps):
            action = self._choose_exploration_action(state)
            next_state, reward, done = self.env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            if done:
                return trajectory, True
        return trajectory, False
