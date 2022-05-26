import neat
import pickle
from run_NEAT import run
import visualize


def run_neat(config_file="config"):

    average_fitness = []

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create a population
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix="nets/Run 4/neat-checkpoint-"))

    # Run for up to 100 generations.
    winner = p.run(run, 1000)

    # Save the best genome :D
    with open("nets/Run 4/winner.pkl", "wb") as file:
        pickle.dump(winner, file)

    # Draw graphs with the simulation data
    visualize.draw_net(config, winner, True, filename="figures/Run 4/winner_net")
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


run_neat("config")
