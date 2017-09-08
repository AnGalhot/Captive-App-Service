package com.fis.secure.communication;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Subsampling1DLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.layers.convolution.Convolution1DLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class AdversariesCryptographyNeuralNet {

	protected static long seed = 12345;
	protected static double dropOut = 0.5;
	public static final int nSamples = 16;
	private static final int batchSize = 16;
	private static final int iterations = 1;
	private static final int nEpochs = 3000;
	public static int MIN_RANGE = 0;
	public static int MAX_RANGE = 3;
	public static final Random rng = new Random(seed);

	private List<double[]> plainTextList = new ArrayList<double[]>();
	private List<double[]> keyTextList = new ArrayList<double[]>();
	private List<double[]> cipherTextList = new ArrayList<double[]>();

	private DenseLayer inputLayer(String name, int nIn, int out, double dropOut, Distribution dist) {
		return new DenseLayer.Builder().name(name).nIn(nIn).nOut(out).dropOut(dropOut).dist(dist).build();
	}

	private DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
		return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
	}

	private ConvolutionLayer firstConvolutionLayer(String name, int out, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 4, 4 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name(name)
				.nOut(out).biasInit(bias).build();

	}

	private ConvolutionLayer firstConvolutionLayer(String name, int inputDepth, int outputDepth, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 4, 4 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name(name)
				.nOut(outputDepth).biasInit(bias).nIn(inputDepth).build();

	}

	private ConvolutionLayer firstConvolutionIDLayer(String name, int inputDepth, int outputDepth, double bias) {
		return new org.deeplearning4j.nn.conf.layers.Convolution1DLayer.Builder(4, 1).name(name).nIn(inputDepth)
				.nOut(outputDepth).biasInit(bias).build();

	}

	private ConvolutionLayer secondConvolutionLayer(String name, int inputDepth, int outputDepth, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 2, 2 }, new int[] { 2, 2 }, new int[] { 1, 1 }).name(name)
				.nOut(outputDepth).biasInit(bias).nIn(inputDepth).build();

	}

	private ConvolutionLayer secondConvolutionIDLayer(String name, int inputDepth, int outputDepth, double bias) {
		return new org.deeplearning4j.nn.conf.layers.Convolution1DLayer.Builder(2, 1).name(name).nIn(inputDepth)
				.nOut(outputDepth).biasInit(bias).build();

	}

	private ConvolutionLayer thirdConvolutionLayer(String name, int inputDepth, int outputDepth, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 1, 1 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name(name)
				.nOut(outputDepth).biasInit(bias).nIn(inputDepth).build();

	}

	private ConvolutionLayer thirdConvolutionIDLayer(String name, int inputDepth, int outputDepth, double bias) {
		return new org.deeplearning4j.nn.conf.layers.Convolution1DLayer.Builder(1, 1).name(name).nIn(inputDepth)
				.nOut(outputDepth).biasInit(bias).build();

	}

	private ConvolutionLayer fourthConvolutionLayer(String name, int inputDepth, int outputDepth, double bias) {
		return new ConvolutionLayer.Builder(new int[] { 1, 1 }, new int[] { 1, 1 }, new int[] { 1, 1 }).name(name)
				.nOut(outputDepth).biasInit(bias).nIn(inputDepth).activation(Activation.TANH).build();

	}

	private ConvolutionLayer fourthConvolution1DLayer(String name, int inputDepth, int outputDepth, double bias) {
		return new org.deeplearning4j.nn.conf.layers.Convolution1DLayer.Builder(1, 1).name(name).nIn(inputDepth)
				.nOut(outputDepth).biasInit(bias).activation(Activation.TANH).build();

	}

	private SubsamplingLayer maxPool(String name, int[] kernel) {
		return new SubsamplingLayer.Builder(kernel, new int[] { 2, 2 }).name(name).build();
	}

	private SubsamplingLayer maxPool(String name, int kernel) {
		return new Subsampling1DLayer.Builder(kernel, 2).name(name).build();
	}

	public static void main(String... args) {
		AdversariesCryptographyNeuralNet a1 = new AdversariesCryptographyNeuralNet();
		AdversariesCryptographyNeuralNet.LegitmateSenderNeuralNet senderNeuralNet = a1.new LegitmateSenderNeuralNet();
		MultiLayerNetwork multiLayerNetwork = senderNeuralNet.constructModel();
		DataSetIterator dsItr = a1.getTrainingData(batchSize, rng);
		for (int i = 0; i < nEpochs; i++) {
			dsItr.reset();
			multiLayerNetwork.fit(dsItr);
		}

		final INDArray testInput = Nd4j.create(new double[] { 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0 },
				new int[] { 1, 12 });
		final INDArray testInput1 = Nd4j.create(new double[] { 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0 },
				new int[] { 1, 12 });
		INDArray out = multiLayerNetwork.output(testInput,false);
		System.out.println(out);
		INDArray out1 = multiLayerNetwork.output(testInput1,false);
		System.out.println(out1);

	}

	private class LegitmateSenderNeuralNet {

		public MultiLayerNetwork constructModel() {
			// MultiLayerConfiguration config = new
			// NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
			NeuralNetConfiguration.Builder networkBuilder = new NeuralNetConfiguration.Builder().seed(seed)
					.iterations(iterations);

			networkBuilder.regularization(true);
			networkBuilder.l2(0.0005);
			networkBuilder.activation(Activation.RELU);
			networkBuilder.learningRate(0.0001);
			networkBuilder.weightInit(WeightInit.XAVIER);
			networkBuilder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
			networkBuilder.updater(Updater.ADAM);
			networkBuilder.momentum(0.9);
			ListBuilder layerBuilder = networkBuilder.list();
			layerBuilder.layer(0, inputLayer("inputLayer", 12, 12, dropOut, new GaussianDistribution(0, 0.005)));
			layerBuilder.layer(1, fullyConnected("FCLayer", 12, 12, dropOut, new GaussianDistribution(0, 0.005)));
			layerBuilder.layer(2, firstConvolutionIDLayer("CNL1", 1, 2, 0));
			// layerBuilder.layer(3, maxPool("pool1", new int[] { 2, 2 }));
			layerBuilder.layer(3, maxPool("pool1", 2));
			layerBuilder.layer(4, secondConvolutionIDLayer("CNL2", 2, 4, 0));
			// layerBuilder.layer(5, maxPool("pool2", new int[] { 2, 2 }));
			layerBuilder.layer(5, maxPool("pool2", 2));
			layerBuilder.layer(6, thirdConvolutionIDLayer("CNL3", 4, 4, 0));
			// layerBuilder.layer(7, maxPool("pool3", new int[] { 2, 2 }));
			layerBuilder.layer(7, maxPool("pool3", 2));
			layerBuilder.layer(8, fourthConvolution1DLayer("CNL4", 4, 1, 0));
			layerBuilder.backprop(true);
			MultiLayerConfiguration config = layerBuilder.pretrain(false).setInputType(InputType.recurrent(12)).build();
			return new MultiLayerNetwork(config);

		}
	}

	private class LegitmateReceiverNeuralNet {

		private MultiLayerNetwork constructModel() {
			// MultiLayerConfiguration config = new
			// NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
			NeuralNetConfiguration.Builder networkBuilder = new NeuralNetConfiguration.Builder().seed(seed)
					.iterations(iterations);

			networkBuilder.regularization(true);
			networkBuilder.l2(0.0005);
			networkBuilder.activation(Activation.RELU);
			networkBuilder.learningRate(0.0001);
			networkBuilder.weightInit(WeightInit.XAVIER);
			networkBuilder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
			networkBuilder.updater(Updater.ADAM);
			networkBuilder.momentum(0.9);
			ListBuilder layerBuilder = networkBuilder.list();
			layerBuilder.layer(0, inputLayer("inputLayer", 32, 32, dropOut, new GaussianDistribution(0, 0.005)));
			layerBuilder.layer(1, fullyConnected("FCLayer", 1024, 1, dropOut, new GaussianDistribution(0, 0.005)));
			layerBuilder.layer(2, firstConvolutionLayer("CNL1", 1, 2, 0));
			layerBuilder.layer(3, maxPool("pool1", new int[] { 2, 2 }));
			layerBuilder.layer(4, secondConvolutionLayer("CNL2", 2, 4, 0));
			layerBuilder.layer(5, maxPool("pool2", new int[] { 2, 2 }));
			layerBuilder.layer(6, thirdConvolutionLayer("CNL3", 4, 4, 0));
			layerBuilder.layer(7, maxPool("pool3", new int[] { 2, 2 }));
			layerBuilder.layer(8, fourthConvolutionLayer("CNL4", 4, 1, 0));
			layerBuilder.backprop(true);
			MultiLayerConfiguration config = layerBuilder.pretrain(false)
					.setInputType(InputType.convolutionalFlat(100, 100, 3)).build();
			return new MultiLayerNetwork(config);

		}
	}

	private class IlegtimateOutsiderNeuralNet {

		private MultiLayerNetwork constructModel() {
			// MultiLayerConfiguration config = new
			// NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
			NeuralNetConfiguration.Builder networkBuilder = new NeuralNetConfiguration.Builder().seed(seed)
					.iterations(iterations);

			networkBuilder.regularization(true);
			networkBuilder.l2(0.0005);
			networkBuilder.activation(Activation.RELU);
			networkBuilder.learningRate(0.0001);
			networkBuilder.weightInit(WeightInit.XAVIER);
			networkBuilder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
			networkBuilder.updater(Updater.ADAM);
			networkBuilder.momentum(0.9);
			ListBuilder layerBuilder = networkBuilder.list();
			layerBuilder.layer(0, inputLayer("inputLayer", 32, 32, dropOut, new GaussianDistribution(0, 0.005)));
			layerBuilder.layer(1, fullyConnected("FCLayer", 512, 1, dropOut, new GaussianDistribution(0, 0.005)));
			layerBuilder.layer(2, firstConvolutionLayer("CNL1", 1, 2, 0));
			layerBuilder.layer(3, maxPool("pool1", new int[] { 2, 2 }));
			layerBuilder.layer(4, secondConvolutionLayer("CNL2", 2, 4, 0));
			layerBuilder.layer(5, maxPool("pool2", new int[] { 2, 2 }));
			layerBuilder.layer(6, thirdConvolutionLayer("CNL3", 4, 4, 0));
			layerBuilder.layer(7, maxPool("pool3", new int[] { 2, 2 }));
			layerBuilder.layer(8, fourthConvolutionLayer("CNL4", 4, 1, 0));
			layerBuilder.backprop(true);
			MultiLayerConfiguration config = layerBuilder.pretrain(false)
					.setInputType(InputType.convolutionalFlat(100, 100, 3)).build();
			return new MultiLayerNetwork(config);

		}
	}

	private DataSetIterator getTrainingData(int batchSize, Random rand) {
		plainTextList.add(new double[] { 1, 1, 0, 0, 0, 0 });
		plainTextList.add(new double[] { 1, 1, 0, 0, 0, 1 });
		plainTextList.add(new double[] { 1, 1, 0, 0, 1, 0 });
		plainTextList.add(new double[] { 1, 1, 0, 0, 1, 1 });
		plainTextList.add(new double[] { 1, 1, 0, 1, 0, 0 });
		plainTextList.add(new double[] { 1, 1, 0, 1, 0, 1 });
		plainTextList.add(new double[] { 1, 1, 0, 1, 1, 0 });
		plainTextList.add(new double[] { 1, 1, 0, 1, 1, 1 });
		plainTextList.add(new double[] { 1, 1, 1, 0, 0, 0 });
		plainTextList.add(new double[] { 1, 1, 1, 0, 0, 1 });
		plainTextList.add(new double[] { 1, 1, 1, 0, 1, 0 });
		plainTextList.add(new double[] { 1, 1, 1, 0, 1, 1 });
		plainTextList.add(new double[] { 0, 0, 0, 0, 0, 1 });
		plainTextList.add(new double[] { 0, 0, 0, 0, 1, 0 });
		plainTextList.add(new double[] { 0, 0, 0, 0, 1, 1 });
		plainTextList.add(new double[] { 0, 0, 0, 1, 0, 0 });

		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });
		keyTextList.add(new double[] { 1, 0, 0, 1, 0, 0 });

		INDArray input = Nd4j.zeros(16, 6);
		int count = 0;
		for (double[] dArray : plainTextList) {
			int index = 0;
			for (double d : dArray) {
				input.putScalar(new int[] { count, index }, d);
				index++;
			}
			count++;
		}

		INDArray keyArray = Nd4j.zeros(16, 6);
		int keyCount = 0;
		for (double[] dArray : keyTextList) {
			int index = 0;
			for (double d : dArray) {
				keyArray.putScalar(new int[] { keyCount, index }, d);
				index++;
			}
			keyCount++;
		}

		cipherTextList.add(new double[] { 1, 1, 1, 1, 1, 1 });
		cipherTextList.add(new double[] { 1, 1, 0, 0, 1, 0 });
		cipherTextList.add(new double[] { 1, 0, 1, 1, 0, 0 });
		cipherTextList.add(new double[] { 1, 1, 1, 0, 1, 0 });
		cipherTextList.add(new double[] { 1, 0, 1, 0, 1, 0 });
		cipherTextList.add(new double[] { 1, 0, 0, 0, 1, 1 });
		cipherTextList.add(new double[] { 1, 1, 1, 0, 0, 0 });
		cipherTextList.add(new double[] { 0, 0, 0, 1, 1, 1 });
		cipherTextList.add(new double[] { 0, 1, 0, 1, 0, 1 });
		cipherTextList.add(new double[] { 1, 1, 0, 0, 1, 1 });
		cipherTextList.add(new double[] { 1, 0, 1, 1, 1, 1 });
		cipherTextList.add(new double[] { 0, 1, 1, 1, 0, 1 });
		cipherTextList.add(new double[] { 0, 0, 0, 0, 1, 0 });
		cipherTextList.add(new double[] { 1, 0, 0, 1, 1, 0 });
		cipherTextList.add(new double[] { 0, 0, 1, 0, 1, 1 });
		cipherTextList.add(new double[] { 0, 1, 1, 0, 1, 0 });

		INDArray labels = Nd4j.zeros(16, 6);
		int labelCount = 0;
		for (double[] dArray : cipherTextList) {
			int index = 0;
			for (double d : dArray) {
				labels.putScalar(new int[] { labelCount, index }, d);
				index++;
			}
			labelCount++;
		}

		INDArray inputNDArray = Nd4j.hstack(input, keyArray);
		DataSet dataSet = new DataSet(inputNDArray, labels);
		List<DataSet> listDataSets = dataSet.asList();
		Collections.shuffle(listDataSets, rng);
		System.out.println("Inside TrainingData method !!!");
		DataSetIterator dsItr = new ListDataSetIterator(listDataSets, batchSize);
		return dsItr;
	}

}