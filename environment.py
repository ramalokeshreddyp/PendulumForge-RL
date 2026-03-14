import gymnasium as gym
import numpy as np
import pymunk
import pygame
import math
from gymnasium import spaces

class DoublePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, reward_type='shaped', render_mode=None, legacy_api=True):
        super(DoublePendulumEnv, self).__init__()
        
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.legacy_api = legacy_api
        
        # Physics constants
        self.dt = 1.0 / 60.0
        self.cart_mass = 1.0
        self.pole1_mass = 0.1
        self.pole2_mass = 0.1
        self.pole1_length = 100.0
        self.pole2_length = 100.0
        self.max_force = 20.0
        
        # Observation space: [cart_x, cart_v, theta1, omega1, theta2, omega2]
        # theta is angle from upright (0 is up)
        high = np.array([
            2.4,                # cart position
            np.finfo(np.float32).max, # cart velocity
            np.pi,              # pole 1 angle
            np.finfo(np.float32).max, # pole 1 angular velocity
            np.pi,              # pole 2 angle
            np.finfo(np.float32).max  # pole 2 angular velocity
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Action space: force applied to cart
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Pymunk and Pygame setup
        self.space = pymunk.Space()
        self.space.gravity = (0, 980)
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.space = pymunk.Space()
        self.space.gravity = (0, 980) # Gravity points down
        
        # Cart
        self.cart_body = pymunk.Body(self.cart_mass, pymunk.moment_for_box(self.cart_mass, (50, 30)))
        self.cart_body.position = (self.screen_width / 2, 300)
        self.cart_shape = pymunk.Poly.create_box(self.cart_body, (50, 30))
        self.cart_shape.elasticity = 0.5
        self.cart_shape.friction = 0.5
        self.space.add(self.cart_body, self.cart_shape)
        
        # Track constraint (GrooveJoint)
        static_body = self.space.static_body
        groove = pymunk.GrooveJoint(static_body, self.cart_body, (0, 300), (self.screen_width, 300), (0, 0))
        self.space.add(groove)
        
        # Pole 1
        self.pole1_body = pymunk.Body(self.pole1_mass, pymunk.moment_for_segment(self.pole1_mass, (0, 0), (0, -self.pole1_length), 5))
        self.pole1_body.position = self.cart_body.position
        self.pole1_shape = pymunk.Segment(self.pole1_body, (0, 0), (0, -self.pole1_length), 5)
        self.space.add(self.pole1_body, self.pole1_shape)
        
        # Joint 1: Cart to Pole 1
        joint1 = pymunk.PivotJoint(self.cart_body, self.pole1_body, (0, 0), (0, 0))
        self.space.add(joint1)
        
        # Pole 2
        self.pole2_body = pymunk.Body(self.pole2_mass, pymunk.moment_for_segment(self.pole2_mass, (0, 0), (0, -self.pole2_length), 5))
        self.pole2_body.position = (self.pole1_body.position[0], self.pole1_body.position[1] - self.pole1_length)
        self.pole2_shape = pymunk.Segment(self.pole2_body, (0, 0), (0, -self.pole2_length), 5)
        self.space.add(self.pole2_body, self.pole2_shape)
        
        # Joint 2: Pole 1 to Pole 2
        joint2 = pymunk.PivotJoint(self.pole1_body, self.pole2_body, (0, -self.pole1_length), (0, 0))
        self.space.add(joint2)
        
        # Add some noise to initial state
        self.pole1_body.angle = self.np_random.uniform(-0.1, 0.1)
        self.pole2_body.angle = self.np_random.uniform(-0.1, 0.1)
        
        obs = self._get_obs()
        if self.legacy_api:
            return obs
        return obs, {}

    def step(self, action):
        # Apply force
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        force = float(action[0]) * self.max_force
        self.cart_body.apply_force_at_local_point((force, 0), (0, 0))
        
        # Step physics
        self.space.step(self.dt)
        
        obs = self._get_obs()
        reward = self._calculate_reward(obs, action)
        
        # Termination conditions
        cart_x = obs[0]
        theta1 = obs[2]
        theta2 = obs[4]
        
        terminated = bool(
            abs(cart_x) > 2.4 or
            abs(theta1) > 0.8 or # ~45 degrees
            abs(theta2) > 0.8
        )
        truncated = False
        done = terminated or truncated
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
        }

        if self.legacy_api:
            return obs, reward, done, info
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Normalize cart position relative to center
        cart_x = (self.cart_body.position.x - self.screen_width / 2) / 100.0
        cart_v = self.cart_body.velocity.x / 100.0
        
        # Pymunk angles are in radians. 0 is pointing right.
        # We want 0 to be UP. In pymunk with gravity (0, 980), UP is -pi/2.
        # But let's just use the body angle relative to vertical.
        theta1 = self.pole1_body.angle
        omega1 = self.pole1_body.angular_velocity
        
        theta2 = self.pole2_body.angle
        omega2 = self.pole2_body.angular_velocity
        
        return np.array([cart_x, cart_v, theta1, omega1, theta2, omega2], dtype=np.float32)

    def _calculate_reward(self, obs, action):
        cart_x, cart_v, theta1, omega1, theta2, omega2 = obs
        
        # Baseline: Upright bonus
        reward = math.cos(theta1) + math.cos(theta2)
        
        if self.reward_type == 'shaped':
            # Center penalty
            reward -= abs(cart_x) * 0.1
            # Velocity penalty (stability)
            reward -= (abs(omega1) + abs(omega2)) * 0.01
            # Action penalty (energy efficiency)
            reward -= (float(action[0])**2) * 0.001
            
        return reward

    def render(self, mode='human'):
        active_mode = self.render_mode if self.render_mode is not None else mode

        if active_mode is None:
            return
            
        if self.screen is None:
            pygame.init()
            if active_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        
        # Draw track
        pygame.draw.line(self.screen, (200, 200, 200), (0, 300), (self.screen_width, 300), 2)
        
        # Draw cart
        cart_pos = self.cart_body.position
        pygame.draw.rect(self.screen, (0, 0, 255), (cart_pos.x - 25, cart_pos.y - 15, 50, 30))
        
        # Draw poles
        def draw_pole(body, length, color):
            start = body.position
            angle = body.angle
            end = (start.x + length * math.sin(angle), start.y - length * math.cos(angle))
            pygame.draw.line(self.screen, color, (start.x, start.y), end, 5)
            return end

        p1_end = draw_pole(self.pole1_body, self.pole1_length, (255, 0, 0))
        draw_pole(self.pole2_body, self.pole2_length, (0, 255, 0))

        if active_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
