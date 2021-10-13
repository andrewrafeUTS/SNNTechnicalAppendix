This code was used for experimentation in Chapter 5 (Evolutionary Experiments with Spiking Neural Networks)

Run on Python 3.8.0. Other versions will work but have not been extensively tested.

PACKAGES NEEDED (can be installed with pip):
- matplotlib
- numpy
- gym

RUNNING THE FILES:
The two files that when run will undergo the experiments are RunCartPole.py and RunLunarLander.py.
These are currently set to the single layer experiments with the network shapes outlined on line 42
and line 24 in their respective files. These can be changed to change the shape of the network.
The [8,8,2] and [14,14,4] are the current network shapes. Taking [8,8,2] as an example, this can be
read as 8 nodes in the input encoder, 8 nodes in the input layer and 2 nodes in the output layer.
The first two values, namely 8,8 need to be the same as these make up the input encoder connections
which are one to one. The experiment was also conducted with multiple layers which to recreate should 
be changed to [8,8,64,2] for CartPole which indicates 8 input decoder nodes, 8 input layer neurons, 64 
hidden layer neurons and 2 output neurons. The multilayer Lunar Lander experiment outlined in Section 5.3
(the main evolutionary experiment) used [14,14,128,4].

The exposure period and decoding method can be then passed as arguments when running the program. For 
example:

	python RunCartPole.py 10 f2freset

This will run the cart pole experiment with whatever network shape is in the file on line 42 with a state
exposure period of 10 using the first-to-fire reset decoding method. The choices of decoding method include:
	- f2f
	- f2freset
	- rate
	- ratereset
The exposure period included can be any positive integer. Just note that the larger the exposure period, the
more processing is needed for each step in the episodes.

When run, this will print out firstly the experiment parameters, then for each generation will print out the
fitness of the best agent and finally when the experiment is finished by reaching the goal, the resultant weights
of the goal achieving agents will be printed.

