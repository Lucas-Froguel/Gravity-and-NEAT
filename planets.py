# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:23:04 2021

@author: Lucas
"""
import pyglet
import numpy as np
from pyglet import shapes


class Planet:
    def __init__(self, mass=100, position=[200, 300], initial_velocity=[20, 30], radius=None,
                 max_radius=30, min_radius=10, color=(255, 250, 0), trajectory_length=300):
        # the position should be given with respect to the center of the screen (use center_x, center_y)
        self.mass = mass
        self.position_n = np.array(position)
        self.position_n_1 = np.array(position)
        if radius:
            self.radius = radius
        else:
            self.radius = np.tanh(np.sqrt(np.log10(mass))) * max_radius - min_radius
        self.v_n = np.array(initial_velocity)
        self.v = np.sqrt(np.sum(np.array(initial_velocity)**2))
        self.forces_n = np.zeros(2)
        self.color = color
        self.a_n = np.zeros(2)
        self.trajectory = []
        self.trajectory_length = trajectory_length

    def draw_planet(self, center_x, center_y):
        shapes.Circle(x=self.position_n[0] + center_x, y=self.position_n[1] + center_y,
                      radius=self.radius, color=self.color).draw()

    def draw_trajectory(self, center_x, center_y):
        batch = pyglet.graphics.Batch()
        lines = []
        for k in range(len(self.trajectory)-1):
            x2, y2 = self.trajectory[k+1] + np.array([center_x, center_y])
            x1, y1 = self.trajectory[k] + np.array([center_x, center_y])
            line = shapes.Line(x1, y1, x2, y2, color=self.color, batch=batch)
            lines.append(line)
        batch.draw()

    def calculate_initial_position(self, dt):
        self.position_n = self.position_n_1 + self.v_n * dt

    def calculate_position(self, dt):
        self.position_n_1 = self.position_n
        self.position_n = self.position_n + self.v_n * dt + self.a_n * dt**2
        self.trajectory.append(self.position_n)
        if len(self.trajectory) > self.trajectory_length:
            self.trajectory.pop(0)

    def calculate_velocity(self, dt):
        self.v_n = (self.position_n - self.position_n_1) / dt

    def calculate_acceleration(self):
        self.a_n = self.forces_n / self.mass




