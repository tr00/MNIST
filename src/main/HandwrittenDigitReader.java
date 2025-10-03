package main;

import java.util.ArrayList;

import net.NeuralNetwork;
import utils.Activation;
import utils.Cost;
import utils.Derivative;
import utils.MNIST;
import utils.Matrix;

public class HandwrittenDigitReader {
	
	private static final int[] train_labels = MNIST.getLabels("res/train-labels.idx1-ubyte");
	private static final ArrayList<int[][]> train_data = MNIST.getImages("res/train-images.idx3-ubyte");
	private static final int[] test_labels = MNIST.getLabels("res/t10k-labels.idx1-ubyte");
	private static final ArrayList<int[][]> test_data = MNIST.getImages("res/t10k-images.idx3-ubyte");

	private NeuralNetwork network;
	
	private final float learningrate;
	
	public static void main(String[] args) {
		//MNIST.test(labels, data);

		HandwrittenDigitReader ai = new HandwrittenDigitReader(Cost.QUADRATIC, .05f);
		
		ai.activate(new Activation[] {Activation.Tanh, Activation.LeakyReLU, Activation.Sigmoid}, 
					new Derivative[] {Derivative.Tanh, Derivative.LeakyReLU, Derivative.Sigmoid});
		ai.initialize();
		float time = ai.train(train_data.size(), 5, 32);
		float prec = ai.test(test_data.size()) * 10;
		System.out.println("Efficiency: " + (prec / time));
	}

	public HandwrittenDigitReader(Cost cost, float learningrate) {
		this.learningrate = learningrate;
		
		this.network = new NeuralNetwork(new int[] {784, 16, 16, 10}, cost);		
	}

	private void activate(Activation[] activation, Derivative[] derivative) {
		network.activate(activation, derivative);
	}
	
	private void initialize() {
		network.initialize();
		System.out.println("\ninitialized the weights and biases of the neural network");
	}
	
	private float train(int cap, int epochs, int batches) {
		ArrayList<Matrix> in = new ArrayList<Matrix>();
		ArrayList<Matrix> res = new ArrayList<Matrix>();
		for(int c = 0; c < cap; c++) {
			double[] tmp = new double[784];
			int p = 0;
			for(int i = 0; i < train_data.get(c).length; i++) {
				for(int j = 0; j < train_data.get(c)[i].length; j++) {
					tmp[p++] = train_data.get(c)[i][j] / 255d;
				}
			}
			in.add(new Matrix(tmp, tmp.length, 1));
			tmp = new double[10];
			for(int i = 0; i < tmp.length; i++) {
				if(i == train_labels[c]) {
					tmp[i] = 1;
					break;
				}
			}
			res.add(new Matrix(tmp, tmp.length, 1));
		}
		System.out.println("fully parsed the training data");
		System.out.println("\nstarting the training now:");
		return network.train(in.toArray(new Matrix[in.size()]), res.toArray(new Matrix[res.size()]), epochs, batches, learningrate);
	}
	
	private float test(int cap) {
		int correct = 0;
		
		System.out.println("\ntesting the neural network now:");
		
		for(int c = 0; c < cap; c++) {
			double[] tmp = new double[784];
			int p = 0;
			for(int i = 0; i < test_data.get(c).length; i++) {
				for(int j = 0; j < test_data.get(c)[i].length; j++) {
					tmp[p++] = test_data.get(c)[i][j] / 255d;
				}
			}
			if(test_labels[c] == Matrix.max(network.process(new Matrix(tmp, tmp.length, 1)))[0]) {
				correct++;
			}		
		}
		System.out.println("Accuracy: " + (correct * 100f / cap) + "%");
		return correct * 100f / cap;
	}

}
