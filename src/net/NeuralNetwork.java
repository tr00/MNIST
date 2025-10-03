package net;

import java.util.Random;

import utils.Activation;
import utils.Cost;
import utils.Derivative;
import utils.MNIST;
import utils.Matrix;

public class NeuralNetwork {
	
	/**
	 * an array of integers in which each number
	 * is the amount of neurons on that layer
	 */
	private final int[] layers;
	
	/**
	 * a function that is applied in order to
	 * activate a neuron
	 */
	private Activation[] activation;
	
	/**
	 * the derivative of the activation function
	 */
	private Derivative[] derivative;
	
	/**
	 * a function that calculates the degree of change
	 * needed in order to improve
	 */
	private Cost cost;
	
	/**
	 * a matrix array containing all biases sorted by layer
	 */
	private Matrix[] biases;
	
	/**
	 * a matrix array containing all weights sorted by layer
	 */
	private Matrix[] weights;
	
	/**
	 * 
	 */
	public NeuralNetwork(int[] sizes, Cost cost) {
		this.layers = sizes;
		this.cost = cost;
		
		this.biases = new Matrix[sizes.length - 1];
		this.weights = new Matrix[sizes.length - 1];	
	}
	
	/**
	 * 
	 */
	public void activate(Activation[] activation, Derivative[] derivative) {
		this.activation = activation;
		this.derivative = derivative;
	}
	
	/**
	 * this method is initializing the weights and biases with normally distributed values
	 */
	public void initialize() {
		Random rand = new Random();
		
		//for every layer of the neural network
		for(int i = 0; i < layers.length - 1; i++) {
			//the biases of each layer are stored in a column vector
			biases[i] = new Matrix(layers[i + 1], 1);
			//the weights of each layer are stored in a matrix
			weights[i] = new Matrix(layers[i + 1], layers[i]);
			
			//for every neuron on the current layer
			for(int j = 0; j < layers[i + 1]; j++) {
				//initializing the bias with a normally distributed value
				biases[i].set(j, 0, rand.nextGaussian());
				
				//for every node of the layer before the current one
				for(int k = 0; k < layers[i]; k++) {
					//initializing the weigths with a normally distributed value relative its layer
					weights[i].set(j, k, rand.nextGaussian() / Math.sqrt(layers[i]));
				}
			}
		}
	}

	/**
	 * processes the outputs of this neural network for a single input column vector
	 */
	public Matrix process(Matrix mat) {
		//for every layer of the network
		for(int i = 0; i < layers.length - 1; i++) {
			Matrix tmp = new Matrix(weights[i].getRows(), 1);
			
			//the dot product of each neuron and its weight stored in a row matrix
			Matrix.dot(weights[i], mat, tmp);
			
			//adding the bias to all neurons of the current layer
			Matrix.add(tmp, biases[i], tmp);
			
			//applying the activation function to the vectorized matrix
			mat = Matrix.vec(tmp, null, activation[i]);
		}
		//return the result in a column vector
		return mat;
	}
	
	/**
	 * back-propagation is the part of the gradient descent algorithm
	 * that calculates the changes that need to be made to the weights and biases
	 */
	public Matrix[][] backpropagate(Matrix data, Matrix res) {
		Matrix[] deltaGradientWeights = new Matrix[layers.length - 1];
		Matrix[] deltaGradientBiases = new Matrix[layers.length - 1];
		Matrix[] weightedSums = new Matrix[layers.length - 1];
		Matrix[] activations = new Matrix[layers.length];
	
		Matrix tmp, last = data;
	
		activations[0] = last;
	
		for(int i = 0; i < layers.length - 1; i++) {
			tmp = Matrix.dot(weights[i], last, null);
	
			Matrix.add(tmp, biases[i], tmp);
			weightedSums[i] = tmp;
			last = Matrix.vec(tmp, null, activation[i]);
			activations[i + 1] = last;
		}
		Matrix err = cost.d(activations[layers.length - 1], res, weightedSums[layers.length - 2], derivative[layers.length - 2]);
		deltaGradientBiases[layers.length - 2] = err;
		deltaGradientWeights[layers.length - 2] = err.multiplyTransposeM(activations[layers.length - 2]);
	
		for(int i = layers.length - 3; i >= 0; i--) {
			tmp = Matrix.vec(weightedSums[i], null, derivative[i]);
			err = weights[i + 1].multiplyTransposeSelf(err);
			Matrix.pro(err, tmp, err);
			deltaGradientBiases[i] = err;
			deltaGradientWeights[i] = err.multiplyTransposeM(activations[i]);
		}
		return new Matrix[][] {deltaGradientWeights, deltaGradientBiases};
	}

	/**
	 * trains the neural network by using the gradient descent algorithm
	 */
	public float train(Matrix[] data, Matrix[] res, int epochs, int batches, float learningrate) {
		float total = 0;
		//trains on all data (from the parameter) multiple times
		for(int e = 0; e < epochs; e++) {
			long time = System.nanoTime();
			
			//the data are getting shuffled so that the learning process
			//can easily escape local minima
			MNIST.shuffle(data, res);
			
			//stores the changes per batch made to the weights and biases
			Matrix[] gradientBiases = new Matrix[layers.length - 1];
			Matrix[] gradientWeights = new Matrix[layers.length - 1];
			
			//initializes all the elements to empty matrices
			for(int i = 0; i < layers.length - 1; i++) {
				gradientBiases[i] = new Matrix(biases[i].getRows(), biases[i].getColumns());
				gradientWeights[i] = new Matrix(weights[i].getRows(), weights[i].getColumns());
			}
			
			//for every data from the input array
			for(int i = 0; i < data.length; i++) {
				
				
				//calculates the changes with back-propagation
				Matrix[][] deltaGradients = backpropagate(data[i], res[i]);
				
				//stores the the changes so they can be applied to
				//the weights and biases after the current batch
				for(int j = 0; j < layers.length - 1; j++) {
					Matrix.add(gradientWeights[j], deltaGradients[0][j], gradientWeights[j]);
					Matrix.add(gradientBiases[j], deltaGradients[1][j], gradientBiases[j]);
				}
				
				//if a batch is complete or the end of the input data is reached
				if(i > 0 && i % batches == 0 || i == data.length - 1) {
					//for every layer of neurons
					for(int j = 0; j < layers.length - 1; j++) {
						
						//scales the changes with the learning rate
						Matrix.scl(gradientWeights[j], gradientWeights[j], learningrate);
						Matrix.scl(gradientBiases[j], gradientBiases[j], learningrate);
						//applies the changes by subtracting a fraction of the difference
						//of the current stats and the estimated stats
						Matrix.sub(weights[j], gradientWeights[j], weights[j]);
						Matrix.sub(biases[j], gradientBiases[j], biases[j]);
						
						//clears the storage for the upcoming batch
						gradientBiases[j].clear();
						gradientWeights[j].clear();
					}
				}
			}
			total += (float) ((System.nanoTime() - time) / 1.0E9);
			System.out.println("Epoch: " + (e + 1) + "/" + epochs + " Time: " + (float) ((System.nanoTime() - time) / 1.0E9) + "sec"); 
		}
		return total;
	}
	
}
