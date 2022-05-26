import neat.nn
import pyglet
from pyglet import shapes
from planets import Planet
from gravity_NEAT import Gravity
from spaceship import Spaceship
import numpy as np


def run(genomes, config):
    window = pyglet.window.Window(1200, 1000)

    label = pyglet.text.Label('Gravity',
                              font_name='Times New Roman',
                              font_size=36,
                              x=100, y=window.height - 50,
                              anchor_x='center', anchor_y='center')

    center_x, center_y = window.width / 2, window.height / 2

    # Define planets
    planet_1 = Planet(position=[0, 0], initial_velocity=[0.0, 0.0], mass=100000)
    planet_2 = Planet(position=[200, 0], color=(0, 0, 255), initial_velocity=[0.0, -50.0], mass=100000)
    planet_3 = Planet(position=[-250, 200], color=(0, 255, 0), initial_velocity=[0., 0.], mass=10000, radius=10)
    planet_4 = Planet(position=[-200, 0], color=(255, 0, 0), initial_velocity=[0., -20.], mass=10000)
    planet_5 = Planet(position=[-300, 300], color=(255, 0, 255), initial_velocity=[40, 0], mass=10000)
    planet_6 = Planet(position=[-300, -300], color=(100, 100, 255), initial_velocity=[0, 50], mass=10000)
    planet_7 = Planet(position=[300, 300], color=(200, 200, 55), initial_velocity=[-30, -20], mass=10000)
    planet_8 = Planet(position=[400, -300], color=(0, 100, 255), initial_velocity=[0, -10], mass=10000)
    planets = [planet_1, planet_2, planet_3, planet_4, planet_5, planet_6, planet_7, planet_8]

    # Define ships and neural nets
    nets = []
    ge = []
    ships = []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        ship = create_ship(center_x, center_y)
        ships.append(ship)
        g.fitness = 0
        ge.append(g)

    # Define gravitational system
    grav = Gravity(planets, center_x=center_x, center_y=center_y)
    grav.ships = ships
    grav.nets = nets
    grav.genomes = ge
    grav.initial_position()

    # Give grav the window
    grav.window = window

    # Add drawing function
    @window.event
    def on_draw():
        window.clear()
        # image.blit(0, 0)
        label.draw()

        # Draw center of screen
        shapes.Circle(x=center_x, y=center_y, radius=3, color=(255, 255, 255)).draw()  # center
        # Draw ships
        for ship in grav.ships:
            if ship:
                ship.draw_spaceship(center_x, center_y)
                ship.draw_trajectory(center_x, center_y)
                # ship.draw_rays(center_x, center_y)
        # Draw planets
        for p in grav.planets:
            p.draw_planet(center_x, center_y)
            p.draw_trajectory(center_x, center_y)

    def update(dt):
        grav.update(dt)

    # Run pyglet window
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()


def create_ship(center_x, center_y):
    pos = get_random_number(2) * np.array([center_x, center_y])
    theta = np.random.random() * (2*np.pi)
    theta_grad = radians_to_degrees(theta)
    ship = Spaceship(position=pos, initial_velocity=40*get_random_number(2))
    ship.theta_grad = theta_grad
    ship.theta_rad = theta
    return ship


def get_random_number(n=1):
    return 2*np.random.random(n) - 1


def radians_to_degrees(val):
    return val * 180 / np.pi


def degrees_to_radians(val):
    return val * np.pi / 180

