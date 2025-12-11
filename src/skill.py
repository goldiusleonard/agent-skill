import numpy as np
from collections import defaultdict


class Skill:
    def __init__(
        self,
        name,
        preconditions,
        action_sequence,
        embedding,
        success_count=0,
    ):
        self.name = name
        self.preconditions = preconditions
        self.action_sequence = action_sequence
        self.embedding = embedding
        self.success_count = success_count
        self.times_used = 0

    def is_applicable(self, state):
        for key, value in self.preconditions.items():
            if state.get(key) != value:
                return False
        return True

    def __repr__(self):
        return (
            f"Skill({self.name}, used={self.times_used}, success={self.success_count})"
        )


class SkillLibrary:
    def __init__(self, embedding_dim=8):
        self.skills = []
        self.embedding_dim = embedding_dim
        self.skill_stats = defaultdict(lambda: {"attempts": 0, "successes": 0})

    def add_skill(self, skill):
        for existing_skill in self.skills:
            if self._similarity(skill.embedding, existing_skill.embedding) > 0.9:
                existing_skill.success_count += 1
                return existing_skill
        self.skills.append(skill)
        return skill

    def retrieve_skills(self, state, query_embedding=None, top_k=3):
        applicable = [s for s in self.skills if s.is_applicable(state)]
        if query_embedding is not None and applicable:
            similarities = [
                self._similarity(query_embedding, s.embedding) for s in applicable
            ]
            sorted_skills = [
                s for _, s in sorted(zip(similarities, applicable), reverse=True)
            ]
            return sorted_skills[:top_k]
        return sorted(
            applicable,
            key=lambda s: s.success_count / max(s.times_used, 1),
            reverse=True,
        )[:top_k]

    def _similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)

    def get_stats(self):
        return {
            "total_skills": len(self.skills),
            "total_uses": sum(s.times_used for s in self.skills),
            "avg_success_rate": np.mean(
                [s.success_count / max(s.times_used, 1) for s in self.skills]
            )
            if self.skills
            else 0,
        }
