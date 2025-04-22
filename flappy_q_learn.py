import gymnasium as gym
import numpy as np
import pygame
import random
import time
import os
import pickle
from gymnasium import spaces

# Game parameters mowa
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PIPE_GAP = 150
PIPE_FREQUENCY = 100  # frames
BIRD_JUMP_VELOCITY = -10
GRAVITY = 1
GAME_SPEED = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995

class Bird:
    def __init__(self):
        self.x = SCREEN_WIDTH // 4
        self.y = SCREEN_HEIGHT // 2
        self.vel = 0
        self.width = 30
        self.height = 30
        self.alive = True
    
    def jump(self):
        self.vel = BIRD_JUMP_VELOCITY
    
    def move(self):
        self.vel += GRAVITY
        self.y += self.vel
        
        # Keep bird within screen bounds
        if self.y < 0:
            self.y = 0
            self.vel = 0
        if self.y + self.height > SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT - self.height
            self.vel = 0
            self.alive = False
    
    def draw(self, screen):
        pygame.draw.rect(screen, YELLOW, (self.x, self.y, self.width, self.height))
    
    def get_mask(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.height = random.randint(100, SCREEN_HEIGHT - PIPE_GAP - 100)
        self.top_pipe = pygame.Rect(self.x, 0, 60, self.height)
        self.bottom_pipe = pygame.Rect(self.x, self.height + PIPE_GAP, 60, SCREEN_HEIGHT)
        self.passed = False
    
    def move(self):
        self.x -= GAME_SPEED
        self.top_pipe.x = self.x
        self.bottom_pipe.x = self.x
    
    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.top_pipe)
        pygame.draw.rect(screen, GREEN, self.bottom_pipe)
    
    def collide(self, bird):
        bird_mask = bird.get_mask()
        return bird_mask.colliderect(self.top_pipe) or bird_mask.colliderect(self.bottom_pipe)
    
    def is_passed(self, bird):
        if not self.passed and self.x + 60 < bird.x:
            self.passed = True
            return True
        return False

class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()
        
        # Define action and observation space
        # Actions: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)
        
        # Observation space: (bird_y, bird_vel, pipe_top_y, pipe_bottom_y, pipe_x)
        # Each value is normalized to [0, 1] range
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird Gym")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
    
    def _get_obs(self):
        # Find closest pipe
        closest_pipe = None
        for pipe in self.pipes:
            if pipe.x + 60 > self.bird.x:
                closest_pipe = pipe
                break
        
        if closest_pipe:
            # Normalize values to [0, 1] range
            obs = np.array([
                self.bird.y / SCREEN_HEIGHT,
                self.bird.vel / 20,  # Normalize velocity to [-1, 1]
                closest_pipe.height / SCREEN_HEIGHT,
                (closest_pipe.height + PIPE_GAP) / SCREEN_HEIGHT,
                closest_pipe.x / SCREEN_WIDTH
            ], dtype=np.float32)
        else:
            # If no pipes, use default values
            obs = np.array([
                self.bird.y / SCREEN_HEIGHT,
                self.bird.vel / 20,
                0.5,  # Default pipe top height
                0.7,  # Default pipe bottom position
                1.0   # Pipe at the far right
            ], dtype=np.float32)
        
        return obs
    
    def _get_info(self):
        return {
            "score": self.score,
            "bird_alive": self.bird.alive
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.frames = 0
        self.last_pipe = 0
        
        # Add initial pipe
        self.pipes.append(Pipe())
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        # Apply action (0 = do nothing, 1 = jump)
        if action == 1:
            self.bird.jump()
        
        # Game logic
        self.bird.move()
        self.frames += 1
        
        # Add new pipe
        if self.frames - self.last_pipe > PIPE_FREQUENCY:
            self.pipes.append(Pipe())
            self.last_pipe = self.frames
        
        # Move pipes and check collisions
        reward = 0.1  # Small positive reward for surviving
        pipe_passed = False
        
        for pipe in self.pipes:
            pipe.move()
            
            # Check if pipe is passed
            if pipe.is_passed(self.bird):
                self.score += 1
                pipe_passed = True
            
            # Check for collision
            if pipe.collide(self.bird):
                self.bird.alive = False
        
        # Remove off-screen pipes
        self.pipes = [pipe for pipe in self.pipes if pipe.x + 60 > 0]
        
        # Calculate reward
        if not self.bird.alive:
            reward = -100  # Large negative reward for dying
            terminated = True
        else:
            terminated = False
            
        if pipe_passed:
            reward += 10  # Reward for passing a pipe
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird Gym")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)
        
        if self.screen is not None:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
            # Draw background
            self.screen.fill(BLACK)
            
            # Draw bird
            self.bird.draw(self.screen)
            
            # Draw pipes
            for pipe in self.pipes:
                pipe.draw(self.screen)
            
            # Draw score
            score_text = self.font.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(score_text, (10, 10))
            
            if self.render_mode == "human":
                pygame.display.update()
                self.clock.tick(self.metadata["render_fps"])
                
            if self.render_mode == "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
                )
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate, discount):
        self.q_table = {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = EPSILON_START
    
    def discretize_state(self, state):
        """Convert continuous state to discrete for Q-table lookup"""
        # Discretize each dimension
        bird_y = int(state[0] * 10)  # 10 bins for y position
        bird_vel = int((state[1] + 1) * 5)  # 10 bins for velocity
        pipe_top = int(state[2] * 10)  # 10 bins for pipe top height
        pipe_bottom = int(state[3] * 10)  # 10 bins for pipe bottom position
        pipe_x = int(state[4] * 10)  # 10 bins for pipe x position
        
        return (bird_y, bird_vel, pipe_top, pipe_bottom, pipe_x)
    
    def get_action(self, state):
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            return self.get_best_action(state)
    
    def get_best_action(self, state):
        # Discretize state for table lookup
        state_key = self.discretize_state(state)
        
        # If state not in Q-table, add it with zeros
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        # Return action with highest Q-value
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        # Discretize states for table lookup
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        # If states not in Q-table, add them with zeros
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_model(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
                return True
        return False

def train_agent(render_every=100):
    env = FlappyBirdEnv()
    agent = QLearningAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=LEARNING_RATE,
        discount=DISCOUNT
    )
    
    # Try to load existing model
    model_file = "flappy_bird_agent_gym.pkl"
    if agent.load_model(model_file):
        print("Loaded existing model")
    else:
        print("Starting with new model")
    
    scores = []
    best_score = 0
    
    for episode in range(EPISODES):
        # Set render mode for visualization every render_every episodes
        if episode % render_every == 0:
            env = FlappyBirdEnv(render_mode="human")
        else:
            env = FlappyBirdEnv()
            
        state, _ = env.reset()
        terminated = False
        episode_score = 0
        total_reward = 0
        
        # Episode loop
        while not terminated:
            action = agent.get_action(state)
            next_state, reward, terminated, _, info = env.step(action)
            
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            
            if info["score"] > episode_score:
                episode_score = info["score"]
        
        # Update best score
        if episode_score > best_score:
            best_score = episode_score
            agent.save_model(model_file)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Store score
        scores.append(episode_score)
        
        # Print progress
        if episode % 10 == 0:
            avg_score = sum(scores[-10:]) / min(10, len(scores))
            print(f"Episode: {episode}, Score: {episode_score}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}, Total Reward: {total_reward:.2f}")
    
    env.close()
    return agent

def play_game(agent, episodes=5):
    env = FlappyBirdEnv(render_mode="human")
    
    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        total_reward = 0
        
        while not terminated:
            # Use the trained policy (no exploration)
            action = agent.get_best_action(state)
            next_state, reward, terminated, _, info = env.step(action)
            state = next_state
            total_reward += reward
            
            # Control game speed for better visualization
            time.sleep(0.01)
        
        print(f"Game {i+1} over! Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # Register the environment
    gym.register(
        id='FlappyBird-v0',
        entry_point='__main__:FlappyBirdEnv',
    )
    
    # Train agent
    agent = train_agent(render_every=500)
    
    # Save final model
    agent.save_model("flappy_bird_agent_final_gym.pkl")
    
    # Play game with trained agent
    play_game(agent)