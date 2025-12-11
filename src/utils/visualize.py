from matplotlib import pyplot as plt


def visualize_training(stats):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(stats["rewards"])
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 1].plot(stats["steps"])
    axes[0, 1].set_title("Steps per Episode")
    axes[1, 0].plot(stats["skills_learned"])
    axes[1, 0].set_title("Skills in Library")
    axes[1, 1].plot(stats["skill_uses"])
    axes[1, 1].set_title("Cumulative Skill Uses")
    plt.tight_layout()
    plt.savefig("skill_learning_stats.png", dpi=150, bbox_inches="tight")
    plt.show()
