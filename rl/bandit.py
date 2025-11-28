import json
import os
import random
import logging
import sys

from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_level=True)]
)
logger = logging.getLogger("bchef.rl")


class BanditPolicy:
    def __init__(self, path="qtable.json", alpha=0.1, epsilon_0=1.0, decay_rate=0.999, min_epsilon=0.1):
        self.path = path
        self.alpha = alpha
        self.epsilon_0 = epsilon_0
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.actions = ["grandma", "gordon", "neutral"]
        self.states = [0, 1]
        logger.info("🎰 RL BanditPolicy initialized")

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

    def choose_action(self, user_id, state, user_data):
        user_id = str(user_id)
        logger.debug(f"🎯 Choosing action for user {user_id}, state {state}")
        if user_id not in self.Q:
            self.Q[user_id] = [[0.0 for _ in self.actions] for _ in self.states]
            self.timesteps[user_id] = 0
            self.epsilons[user_id] = self.epsilon_0

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

    def _save(self):
        data = {
            "Q": self.Q,
            "timesteps": self.timesteps,
            "epsilons": self.epsilons
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)