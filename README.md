# Gravity and NEAT

Project to simulate a gravitational system and use NEAT to train a spaceship to naviage.


## Gravity and Dynamics

The gravitational interaction between the bodies is given by Newton's law of gravity:
$$ F = -G \frac{Mm}{r^2} $$
However, we do use the vector form of the equation. The motion integration is simply verlet's integration, which is a fancy name for a second-order taylor expansion of the position:
$$ x_n = x_{n-1} + v_n\Delta t + \frac{a_n\Delta t^2}{2} $$
where the velocity is given by $\Delta x_{n-1}/\Delta t$ at each frame. 

The ship's acceleration is determined in two parts: the pull it feels from the planets plus its thrust. The neural net only controls the thrust and must learn
to deal with the other component of the acceleration.


## NEAT

NEAT stands for Neural Evolution of Augmenting Topologies, which is a machine-learning technique used to build not only the weights, but also 
the structure of the neural network that bests solves the problem. 

In this case, I used the python-implementation of NEAT (https://neat-python.readthedocs.io/en/latest/). I plan to do my own in the future, but I am quite happy
with this one. 

I used for inputs 17 values: distance to the center, distances to each wall (4), velocity (2), vision rays (10). Each vision ray is a ray cast to
a long distance that return a -1 if it does not touch anything and returns the distance to a planet, if it gets to touch one. 

The cost function was extremely complicated and probably still is not the best. I used a combination of time alive, acceleration intensity (in order to stimulate
movement,as the ships were learning to do nothing prior to this), avoiding planets and avoiding walls. 

## Results

I trained for 1000 generations and the best results was the following:


https://user-images.githubusercontent.com/77543666/170560997-c94738be-e04d-463e-9bef-2abd158a84cf.mp4

Apparently, the strategy it learned is to try to remain in the same position, which is not bad. 

I also gave a "naive" brain to the ship, where it just tries to accelerate in the direction of the center independetly of everything else (and it didn't work well,
but it was cool to see):


https://user-images.githubusercontent.com/77543666/170561177-40da6d46-2b38-4bf4-ac1a-0b4d2323261b.mp4



