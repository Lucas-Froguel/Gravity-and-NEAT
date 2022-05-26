import pyglet
import numpy as np
from pyglet import sprite, shapes
import math


class Spaceship:
    def __init__(self, mass=100, position=[200, 300], initial_velocity=[-20, -30], trajectory_length=300):
        # the position should be given with respect to the center of the screen (use center_x, center_y)
        # Dynamical Variables
        self.mass = mass
        self.position_n = np.array(position)
        self.position_n_1 = np.array(position)
        self.v_n = np.array(initial_velocity)
        self.v = np.sqrt(np.sum(np.array(initial_velocity) ** 2))
        self.forces_n = np.zeros(2)
        self.a_n = np.zeros(2)
        self.thrust = np.ones(2)

        # Center of the screen
        self.center_x = 0
        self.center_y = 0
        self.center = 0

        # Trajectory
        self.trajectory = []
        self.trajectory_length = trajectory_length

        # Image and Sprite
        self.image = pyglet.image.load("sprites/spaceship.jpg")
        self.image.anchor_x = int(self.image.width / 2)
        self.image.anchor_y = int(self.image.height / 2)
        self.ship = sprite.Sprite(self.image, 0, 0)
        self.image_x = 0
        self.image_y = 0
        self.image_position = np.zeros(2)
        self.theta_grad = 0
        self.theta_rad = 0
        self.bounding_box = None

        # Rays / Vision
        self.n_rays = 10
        self.rays = np.zeros((self.n_rays, 2))
        # vision guards the coordinates of the objects colliding with the rays and
        # bools whether there is something or not
        self.vision_bool = np.zeros(self.n_rays)
        self.vision = np.zeros(self.n_rays)

    def draw_spaceship(self, center_x, center_y):
        self.calculate_rotation()
        x = self.position_n[0] + center_x
        y = self.position_n[1] + center_y

        if math.isnan(x):
            x = 2000
        if math.isnan(y):
            y = 2000
        if math.isnan(self.theta_grad):
            self.theta_grad = 0
        if math.isnan(x) or math.isnan(y) or math.isnan(self.theta_grad):
            print(x, y, self.theta_grad)
        self.ship.update(scale=0.1, rotation=self.theta_grad, x=x, y=y)
        self.ship.draw()
        if not self.image_x:
            self.image_x = self.ship.width / 2
            self.image_y = self.ship.height / 2
            self.image_position = np.array([self.image_x, self.image_y])
            self.bounding_box = shapes.Circle(x=x, y=y, radius=min(self.image_x, self.image_y))
            self.center_x = center_x
            self.center_y = center_y
            self.center = np.array([center_x, center_y])

    def draw_trajectory(self, center_x, center_y):
        batch = pyglet.graphics.Batch()
        lines = []
        for k in range(len(self.trajectory) - 1):
            center = np.array([center_x, center_y])
            x2, y2 = self.trajectory[k + 1] + center
            x1, y1 = self.trajectory[k] + center
            line = shapes.Line(x1, y1, x2, y2, color=(0, 254, 0), batch=batch)
            lines.append(line)
        batch.draw()

    def draw_rays(self, center_x, center_y):
        batch = pyglet.graphics.Batch()
        rays = []
        for k in range(self.n_rays):
            center = np.array([center_x, center_y])
            x2, y2 = self.rays[k] + center
            x1, y1 = self.position_n + center
            color = (250, 250, 250)
            if self.vision_bool[k]:
                color = (250, 0, 0)
            ray = shapes.Line(x1, y1, x2, y2, color=color, batch=batch)
            rays.append(ray)
        batch.draw()

    def calculate_initial_position(self, dt):
        self.position_n = self.position_n_1 + self.v_n * dt

    def calculate_position(self, dt):
        self.position_n_1 = self.position_n
        self.position_n = self.position_n + self.v_n * dt + self.a_n * dt ** 2
        self.trajectory.append(self.position_n)
        if len(self.trajectory) > self.trajectory_length:
            self.trajectory.pop(0)

    def calculate_velocity(self, dt):
        self.v_n = (self.position_n - self.position_n_1) / dt

    def calculate_acceleration(self):
        self.a_n = self.forces_n / self.mass + self.thrust

    def calculate_rotation(self):
        theta = np.arccos(self.thrust[1] / distance_points(self.thrust, 0))
        self.theta_grad = radians_to_degrees(theta)
        # if the vector is in another domain, we need values of theta_grad in the [pi, 2pi] region
        if self.thrust[0] < 0:
            if self.thrust[1] < 0:
                t = 180 - self.theta_grad
                self.theta_grad += 2 * t
            else:
                self.theta_grad = 360 - self.theta_grad
        self.theta_rad = degrees_to_radians(self.theta_grad)

    def control_thrust(self, center_x, center_y):
        # First try: make it go to the center of the screen with the thrust
        vec = - self.position_n / np.array([center_x, center_y])
        self.thrust = 100 * vec

    def create_rays(self, center_x, center_y):
        intervals = np.linspace(
            self.theta_rad, 2 * np.pi + self.theta_rad - self.theta_rad / self.n_rays, num=self.n_rays
        )
        for k in range(self.n_rays):
            theta = intervals[k]
            ray = self.position_n + 2*np.array([center_x, center_y]) * np.array([np.sin(theta), np.cos(theta)])
            self.rays[k] = ray

    def check_rays(self, planets):
        for k in range(self.n_rays):
            ray = self.rays[k]
            self.vision_bool[k] = 0
            # -1 represents (to the neural net) that there is nothing (hopefully!)
            self.vision[k] = -1
            for planet in planets:
                x0, y0 = planet.position_n
                x1, y1 = self.position_n
                x2, y2 = ray
                d = np.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if d < planet.radius:
                    self.vision_bool[k] = 1
                    self.vision[k] = distance_points(planet.position_n, self.position_n)
                    break

    def calculate_distance_to_border(self):
        x, y = self.position_n
        x_right = self.center_x - x
        x_left = x - self.center_x
        y_up = self.center_y - y
        y_down = y - self.center_y

        return x_right, x_left, y_up, y_down

    def get_neural_net_inputs(self):
        d = distance_points(self.position_n, 0)
        x_right, x_left, y_up, y_down = self.calculate_distance_to_border()
        return d, x_right, x_left, y_up, y_down, self.v_n[0], self.v_n[1],\
               self.vision[0], self.vision[1], self.vision[2], self.vision[3], \
               self.vision[4], self.vision[5], self.vision[6], self.vision[7], self.vision[8], self.vision[9]


def distance_points(p1, p2):
    d = np.sqrt(np.sum((p1 - p2) ** 2))
    return d


def radians_to_degrees(val):
    return val * 180 / np.pi


def degrees_to_radians(val):
    return val * np.pi / 180
