
import neat
import pickle
from run_NEAT import run


def replay_genome(config_path, genome_path="winner.pkl"):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Call game with only the loaded genome
    run(genomes, config)


def replay_checkpoint(config_path, genome_path="winner.pkl"):
    checkpoint = neat.Checkpointer(5, filename_prefix="nets/Run 4/neat-checkpoint-")

    pop = checkpoint.restore_checkpoint("nets/Run 4/neat-checkpoint-590")
    winner = pop.run(run, 10)
    # Save the best genome :D
    with open("nets/Run 4/winner2.pkl", "wb") as file:
        pickle.dump(winner, file)


replay_genome("config", genome_path="nets/Run 4/winner2.pkl")
# replay_checkpoint("config", genome_path="nets/Run 4/winner.pkl")
