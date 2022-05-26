import copy

import numpy as np
from planets import Planet
from spaceship import Spaceship
import time


class Gravity:
    def __init__(self, planets, center_x=0, center_y=0, ships=None, genomes=None, nets=None):
        self.center_x, self.center_y = center_x, center_y
        self.center = np.array([center_x, center_y])
        self.G = 6.67 * 10 ** (0)
        self.forces_planets = np.zeros((len(planets), len(planets), 2))
        self.planets = planets
        self.e = 1
        self.ships = ships
        self.genomes = genomes
        self.nets = nets
        self.choices = range(255)
        self.initial_time = time.time()
        self.window = None

    def calculate_forces_planets(self):
        self.forces_planets = np.zeros((len(self.planets), len(self.planets), 2))
        i, j = 0, 0
        for planet in self.planets:
            j = 0
            for p in self.planets:
                if i < j:
                    r = np.sqrt(np.sum((planet.position_n - p.position_n)**2))
                    if r == 0.:
                        r = np.inf
                    F = - ((self.G * planet.mass * p.mass) / r ** 3) * (planet.position_n - p.position_n)
                    self.forces_planets[i, j] = F
                elif j < i:
                    self.forces_planets[i, j] = - self.forces_planets[j, i]
                j += 1
            i += 1

    def position_planets(self, dt=1/60):
        i = 0
        for p in self.planets:
            F = np.sum(self.forces_planets[i], axis=0)
            p.forces_n = F
            p.calculate_acceleration()
            p.calculate_velocity(dt)
            p.calculate_position(dt)
            i += 1

    def check_collision_planets(self):
        i, j = 0, 0
        for planet in self.planets:
            j = 0
            for p in self.planets:
                if j == i:
                    break
                p1 = planet.position_n
                p2 = p.position_n
                dis = distance_points(p1, p2)
                min_distance = planet.radius + p.radius
                if dis < min_distance:
                    # if a collision happens, we return the planets to a valid position along the collision axis
                    axis_vec = (p1 - p2)/dis
                    delta = min_distance - dis
                    planet.position_n += axis_vec * self.e * delta
                    p.position_n -= axis_vec * self.e * delta
                j += 1
            i += 1

    def check_in_screen_planets(self, dt=1/60):
        k = 0
        for p in range(len(self.planets)):
            p -= k
            px, py = self.planets[p].position_n
            if self.center_x < px or px < -self.center_x or self.center_y < py or py < -self.center_y:
                k += 1
                self.planets.pop(p)
                self.create_planet(dt)

    def position_ship(self, dt=1/60):
        for s in range(len(self.ships)):
            ship = self.ships[s]
            val = self.nets[s].activate(ship.get_neural_net_inputs())
            ship.thrust = 50*np.array(val)
            if all(ship.thrust):
                self.genomes[s].fitness += 10
            else:
                if not all(ship.thrust):
                    self.genomes[s].fitness -= 10
                else:
                    self.genomes[s].fitness -= 5
            ship.calculate_acceleration()
            ship.calculate_velocity(dt)
            ship.calculate_position(dt)

    def calculate_forces_ship(self):
        for ship in self.ships:
            F_total = np.zeros(2)
            for p in self.planets:
                # r = np.sqrt(np.sum((ship.position_n + ship.image_position - p.position_n) ** 2))
                r = np.sqrt(np.sum((ship.position_n - p.position_n) ** 2))
                if r == 0.:
                    r = np.inf
                # F = - ((self.G * ship.mass * p.mass) / r ** 3)*(ship.position_n + ship.image_position - p.position_n)
                F = - ((self.G * ship.mass * p.mass) / r ** 3) * (ship.position_n - p.position_n)
                F_total += F
            ship.forces_n = F_total

    def check_collision_ship_with_planets(self):
        k = 0
        for s in range(len(self.ships)):
            s -= k
            for planet in self.planets:
                # we can add planet.radius for some more difficulty
                if distance_points(self.ships[s].position_n, planet.position_n) < self.ships[s].bounding_box.radius:
                    # We kill the ship and decrease its fitness
                    self.genomes[s].fitness -= 30
                    self.ships[s].ship.delete()
                    self.ships.pop(s)
                    self.genomes.pop(s)
                    self.nets.pop(s)
                    k += 1
                    break

    def check_in_screen_ship(self):
        k = 0
        for s in range(len(self.ships)):
            s -= k
            px, py = self.ships[s].position_n
            if self.center_x < px or px < -self.center_x or self.center_y < py or py < -self.center_y:
                self.genomes[s].fitness -= 50
                self.ships[s].ship.delete()
                self.ships.pop(s)
                self.genomes.pop(s)
                self.nets.pop(s)
                k += 1

    def cast_rays_ship(self):
        for ship in self.ships:
            ship.create_rays(self.center_x, self.center_y)
            ship.check_rays(self.planets)

    def calculate_fitness_ship(self):
        current_time = time.time()
        for s in range(len(self.ships)):
            ship = self.ships[s]
            val = 2*(current_time - self.initial_time)
            val -= 0.05 * distance_points(ship.position_n, self.center) / self.center_x
            # We favor ships that are accelerating, because it is less boring :D
            val += np.sum(np.abs(self.ships[s].thrust)) / 10
            self.genomes[s].fitness = val

    def initial_position(self, dt=1/60):
        if self.ships:
            for ship in self.ships:
                ship.calculate_initial_position(dt)
        for p in self.planets:
            p.calculate_initial_position(dt)

    def create_planet(self, dt=1/60):
        color = tuple(np.random.choice(self.choices, 3))
        pos = get_random_number(2) * np.array([self.center_x, self.center_y])
        planet = Planet(position=pos, color=color,
                        mass=100000*np.random.random(), initial_velocity=50*get_random_number(2))
        planet.calculate_initial_position(dt)
        self.planets.append(planet)

    def update(self, dt):
        if len(self.ships):
            self.calculate_fitness_ship()
            self.calculate_forces_ship()
            self.position_ship(dt=dt)
            self.cast_rays_ship()
            self.check_collision_ship_with_planets()
            self.check_in_screen_ship()
        else:
            self.window.close()
        self.calculate_forces_planets()
        self.position_planets(dt=dt)
        self.check_collision_planets()
        self.check_in_screen_planets(dt)


def distance_points(p1, p2):
    d = np.sqrt(np.sum((p1-p2)**2))
    return d


def get_random_number(n=1):
    return 2*np.random.random(n) - 1
