import json
import os
import random
import logging
import sys
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_level=True)]
)
logger = logging.getLogger("bchef.rl")


class BanditPolicy:
    def __init__(self, path="qtable.json", alpha=0.1, epsilon_0=1.0, decay_rate=0.999, min_epsilon=0.1, visualize_user_id=None):
        self.path = path
        self.alpha = alpha
        self.epsilon_0 = epsilon_0
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.visualize_user_id = str(visualize_user_id) if visualize_user_id else None  # None = all users
        self.actions = ["grandma", "gordon", "neutral"]
        self.states = [0, 1]
        logger.info("🎰 RL BanditPolicy initialized")

        os.makedirs("qtable_plots", exist_ok=True)

        # Load or init Q-table (per-user dict)
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                self.Q = data["Q"]  # {user_id: [[q_gma, q_gor, q_neu] for s in [0,1]]}
                self.timesteps = data["timesteps"]  # {user_id: t}
                self.epsilons = data["epsilons"]  # {user_id: epsilon}
                logger.info(f"📊 Loaded Q-table from {self.path}")
        else:
            self.Q = {}
            self.timesteps = {}
            self.epsilons = {}
            self._save()
            logger.info("🆕 Created new Q-table")

        self.update_counters = {user: 0 for user in self.Q.keys()}

    def choose_action(self, user_id, state, user_data):
        user_id = str(user_id)
        logger.debug(f"🎯 Choosing action for user {user_id}, state {state}")
        if user_id not in self.Q:
            self.Q[user_id] = [[0.0 for _ in self.actions] for _ in self.states]
            self.timesteps[user_id] = 0
            self.epsilons[user_id] = self.epsilon_0
            self.update_counters[user_id] = 0

            initial = user_data.get(int(user_id), {}).get("initial_tone", "neutral")
            if initial in self.actions:
                return initial

        if state not in self.states:
            raise ValueError(f"Invalid state: {state}")

        self.timesteps[user_id] += 1
        self.epsilons[user_id] = max(self.min_epsilon, self.epsilon_0 * (self.decay_rate ** self.timesteps[user_id]))
        self._save()

        if random.random() < self.epsilons[user_id]:
            return random.choice(self.actions)
        else:
            action_idx = self.Q[user_id][state].index(max(self.Q[user_id][state]))
            logger.debug(f"🤖 Selected tone: {self.actions[action_idx]} for user {user_id}")
            return self.actions[action_idx]

    def update(self, user_id, state, action, reward):
        user_id = str(user_id)
        logger.info(f"🔄 Updating Q-table: user {user_id}, state {state}, action {action}, reward {reward}")
        if user_id not in self.Q:
            raise ValueError(f"No Q for user {user_id}")

        if state not in self.states:
            raise ValueError(f"Invalid state: {state}")
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        action_idx = self.actions.index(action)
        old_q = self.Q[user_id][state][action_idx]
        self.Q[user_id][state][action_idx] = old_q + self.alpha * (reward - old_q)
        self._save()

        self._visualize_qtable(user_id)

    def _save(self):
        data = {
            "Q": self.Q,
            "timesteps": self.timesteps,
            "epsilons": self.epsilons
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)

    def _visualize_qtable(self, user_id):
        if self.visualize_user_id and self.visualize_user_id != user_id:
            return

        q_values = self.Q[user_id]
        actions = self.actions
        states = ["State 0 (incorrect)", "State 1 (correct)"]

        df = pd.DataFrame(q_values, columns=actions, index=states)
        logger.info(f"Q-Table for User {user_id}:\n{df.to_string()}")

        user_folder = os.path.join("qtable_plots", f"user_{user_id}")
        os.makedirs(user_folder, exist_ok=True)

        self.update_counters[user_id] = self.update_counters.get(user_id, 0) + 1
        frame_num = self.update_counters[user_id]
        png_path = os.path.join(user_folder, f"update_{frame_num:03d}.png")

        plt.figure(figsize=(8, 5))
        df.plot(kind="bar", title=f"Q-Values for User {user_id} | Update #{frame_num}")
        plt.ylabel("Q-Value")
        plt.xlabel("State")
        plt.xticks(rotation=0)
        plt.legend(title="Actions")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

        logger.info(f"Saved Q-plot: {png_path}")

        self._create_gif(user_id, user_folder)

    def _create_gif(self, user_id, user_folder):
        png_files = sorted([f for f in os.listdir(user_folder) if f.endswith('.png')])
        if not png_files:
            return

        images = []
        for png in png_files:
            images.append(imageio.imread(os.path.join(user_folder, png)))

        gif_path = os.path.join(user_folder, f"qtable_animation_{user_id}.gif")
        imageio.mimsave(gif_path, images, fps=2, loop=0)

        logger.info(f"Updated GIF: {gif_path}")