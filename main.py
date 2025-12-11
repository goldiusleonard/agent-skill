from src.grid_world import GridWorld
from src.memory_agent import ProceduralMemoryAgent
from src.utils.visualize import visualize_training


if __name__ == "__main__":
    print("=== Procedural Memory Agent Demo ===\n")
    env = GridWorld(size=5)
    agent = ProceduralMemoryAgent(env)
    print("Training agent to learn reusable skills...\n")
    stats = agent.train(episodes=15)
    print("\n=== Learned Skills ===")
    for skill in agent.skill_library.skills:
        print(
            f"{skill.name}: {len(skill.action_sequence)} actions, used {skill.times_used} times, {skill.success_count} successes"
        )
    lib_stats = agent.skill_library.get_stats()
    print("\n=== Library Statistics ===")
    print(f"Total skills: {lib_stats['total_skills']}")
    print(f"Total skill uses: {lib_stats['total_uses']}")
    print(f"Avg success rate: {lib_stats['avg_success_rate']:.2%}")
    visualize_training(stats)
    print("\n✓ Skill learning complete! Check the visualization above.")
