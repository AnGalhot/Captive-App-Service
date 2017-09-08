package com.fis.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ANCNeuralNet {

	private static final Logger log = LoggerFactory.getLogger(ANCNeuralNet.class);
	protected static long seed = 12345;
	public static final Random rng = new Random(seed);

	public static void main(String[] args) throws Exception {
		int nChannels = 1; // Number of input channels
		int outputNum = 10; // The number of possible outcomes
		int batchSize = 64; // Test batch size
		int nEpochs = 1; // Number of training epochs
		int iterations = 1; // Number of training iterations
		int seed = 123; //

		/*
		 * Create an iterator using the batch size for one iteration
		 */
		log.info("Load data....");
		// DataSetIterator mnistTrain = getTrainingData();
		DataSetIterator mnistTrain = load("D:/Backup/TEMP/Keystore/ABC.txt", "  ");
		// DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,
		// true, 12345);
		// DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,
		// 12345);

		/*
		 * PrintWriter writer = new PrintWriter(new
		 * File("D:/Backup/TEMP/Keystore/traindata.predict")); int total = 0;
		 * while (mnistTrain.hasNext()) { DataSet ds = mnistTrain.next();
		 * INDArray input = ds.getFeatureMatrix(); write(input,writer); total++;
		 * if(total == 13) { break; } } writer.flush(); writer.close(); int
		 * total = 0; PrintWriter writer1 = new PrintWriter(new
		 * File("D:/Backup/TEMP/Keystore/labeldata.predict")); while
		 * (mnistTrain.hasNext()) { DataSet ds = mnistTrain.next(); INDArray
		 * input = ds.getLabels(); write(input,writer1); total++; if(total ==
		 * 13) { break; } } writer1.flush(); writer1.close();
		 */
		/*
		 * DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,
		 * false, 12345);
		 * 
		 * while (mnistTest.hasNext()) { DataSet ds = mnistTest.next(); INDArray
		 * input = ds.getFeatureMatrix();
		 * 
		 * }
		 */

		/*
		 * Construct the neural network
		 */
		log.info("Build model....");

		// learning rate schedule in the form of <Iteration #, Learning Rate>
		Map<Integer, Double> lrSchedule = new HashMap<>();
		lrSchedule.put(0, 0.01);
		lrSchedule.put(1000, 0.005);
		lrSchedule.put(3000, 0.001);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations) // Training
																												// iterations
																												// as
																												// above
				.regularization(true).l2(0.0005)
				/*
				 * Uncomment the following for learning decay and bias
				 */
				.learningRate(.01)// .biasLearningRate(0.02)
				/*
				 * Alternatively, you can use a learning rate schedule.
				 * 
				 * NOTE: this LR schedule defined here overrides the rate set in
				 * .learningRate(). Also, if you're using the Transfer Learning
				 * API, this same override will carry over to your new model
				 * configuration.
				 */
				.learningRateDecayPolicy(LearningRatePolicy.Schedule).learningRateSchedule(lrSchedule)
				/*
				 * Below is an example of using inverse policy rate decay for
				 * learning rate
				 */
				// .learningRateDecayPolicy(LearningRatePolicy.Inverse)
				// .lrPolicyDecayRate(0.001)
				// .lrPolicyPower(0.75)
				.weightInit(WeightInit.XAVIER).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS) // To configure: .updater(new
											// Nesterovs(0.9))
				.list()
				.layer(0,
						new ConvolutionLayer.Builder(5, 5)
								// nIn and nOut specify depth. nIn here is the
								// nChannels and nOut is the number of filters
								// to be applied
								.nIn(nChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(2,
						new ConvolutionLayer.Builder(5, 5)
								// Note that nIn need not be specified in later
								// layers
								.stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
				.layer(3,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
				.layer(5,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
								.activation(Activation.SOFTMAX).build())
				.setInputType(InputType.convolutionalFlat(28, 28, 1)) // See
																		// note
																		// below
				.backprop(true).pretrain(false).build();

		/*
		 * Regarding the .setInputType(InputType.convolutionalFlat(28,28,1))
		 * line: This does a few things. (a) It adds preprocessors, which handle
		 * things like the transition between the convolutional/subsampling
		 * layers and the dense layer (b) Does some additional configuration
		 * validation (c) Where necessary, sets the nIn (number of input
		 * neurons, or input depth in the case of CNNs) values for each layer
		 * based on the size of the previous layer (but it won't override values
		 * manually set by the user)
		 * 
		 * InputTypes can be used with other layer types too (RNNs, MLPs etc)
		 * not just CNNs. For normal images (when using ImageRecordReader) use
		 * InputType.convolutional(height,width,depth). MNIST record reader is a
		 * special case, that outputs 28x28 pixel grayscale (nChannels=1)
		 * images, in a "flattened" row vector format (i.e., 1x784 vectors),
		 * hence the "convolutionalFlat" input type used here.
		 */

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		log.info("Train model....");
		model.setListeners(new ScoreIterationListener(1));
		for (int i = 0; i < nEpochs; i++) {
			model.fit(mnistTrain);
			log.info("*** Completed epoch {} ***", i);

			/*
			 * log.info("Evaluate model...."); Evaluation eval = new
			 * Evaluation(outputNum); while (mnistTest.hasNext()) { DataSet ds =
			 * mnistTest.next(); INDArray output =
			 * model.output(ds.getFeatureMatrix(), false);
			 * eval.eval(ds.getLabels(), output);
			 * 
			 * } log.info(eval.stats()); mnistTest.reset();
			 */
		}
		/*
		 * float[] test = buildFeatureVec("110000100100"); float[][] testData =
		 * new float[1][0]; testData[0] = test; INDArray testfeature =
		 * Nd4j.create(testData); INDArray out = model.output(testfeature,
		 * false); System.out.println(out);
		 */
		/*
		 * inal INDArray testinput = Nd4j.create(new float[]
		 * {0,0,0,1,0,0,1,0,0,1,0,0}, new int[] { 1, 12 }); final INDArray
		 * testinput1 = Nd4j.create(new double[] {0,0,0,0,1,1,1,0,0,1,0,0}, new
		 * int[] { 1, 12 }); INDArray out = net.output(testinput, false);
		 * System.out.println(out); INDArray out1 = net.output(testinput1,
		 * false); System.out.println(out1);
		 */
		mnistTrain.reset();
		INDArray testData = null;
		int count = 0;

		while (mnistTrain.hasNext()) {
			count++;
			if (count < 30) {
				continue;
			}

			DataSet ds = mnistTrain.next();
			testData = ds.getFeatureMatrix();
			break;
		}
		System.out.println(testData);
		INDArray result = model.output(testData);
		System.out.println(result);

		// String testDta = "00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.32 1.00 0.99 0.48 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.32 0.95 0.99 0.67 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.24 1.00 0.99 0.80 0.08 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.24 0.80 0.99 0.67 0.08 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.68 0.99 0.96 0.32 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.08 0.56 0.99 0.83 0.32 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.20 0.91 1.00 0.51 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.91 0.99 0.91 0.12 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.56 1.00 0.99 0.16 0.00 0.00 0.00 0.00
		// String testDta="0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.25 0.75 0.87 1.00 0.87 0.75 0.25 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.37
		// 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.37 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.87
		// 1.00 0.37 0.12 0.00 0.12 0.50 1.00 0.87 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00
		// 1.00 0.50 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.62
		// 1.00 1.00 1.00 0.62 0.37 0.25 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.50 0.87 1.00 1.00 1.00 1.00 0.87 0.25 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.25 0.50 0.75 1.00 1.00 0.75 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.12 1.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.75
		// 1.00 0.37 0.00 0.00 0.12 0.50 1.00 0.75 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.25
		// 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.25 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.25 0.75 0.87 1.00 0.87 0.62 0.25 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00";
		String testDta = "0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.77  0.67  0.04  0.00  0.00  0.00  0.05  0.55  0.19  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.07  0.87  0.91  0.24  0.00  0.00  0.03  0.88  0.89  0.05  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.05  0.83  1.00  0.15  0.00  0.00  0.38  1.00  0.42  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.20  0.97  0.72  0.01  0.00  0.09  0.86  0.72  0.01  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.12  0.86  0.99  0.24  0.00  0.12  0.62  0.84  0.09  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.35  0.97  0.53  0.00  0.00  0.54  0.00  0.61  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.04  0.77  0.98  0.21  0.00  0.07  0.83  0.00  0.18  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.46  0.99  0.45  0.00  0.00  0.46  0.99  0.57  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.33  0.97  0.72  0.04  0.00  0.10  0.84  0.88  0.24  0.06  0.36  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.45  1.00  0.62  0.00  0.00  0.00  0.91  0.96  0.19  0.00  0.53  0.38  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.77  0.95  0.12  0.00  0.00  0.09  1.00  0.83  0.00  0.18  0.91  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.34  0.96  0.78  0.08  0.08  0.08  0.58  1.00  0.77  0.84  0.85  0.39  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.10  0.94  0.99  1.00  0.99  0.99  0.99  0.99  1.00  0.85  0.56  0.26  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.45  0.99  0.58  0.46  0.46  0.72  0.98  0.99  0.51  0.10  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.34  0.25  0.05  0.00  0.00  0.24  0.97  0.69  0.09  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.02  0.61  1.00  0.23  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.14  0.99  0.80  0.04  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.07  0.74  0.89  0.05  0.00  0.00  0.00  0.00  0.00  0.00  0.50  0.50  0.50  0.50  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.23  0.99  0.80  0.03  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.38  0.96  0.27  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00";
		// String testDta="0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.84 0.99 0.78 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.62 0.99 0.78 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.35 0.88 0.58 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.84 0.99 0.26 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.10 0.87 0.99 0.26 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.48 0.99 0.93 0.20 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.29 0.96 0.99 0.73 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.42 0.99 0.99 0.73 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.05 0.73 0.99 0.97 0.42 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.99 0.99 0.99 0.40 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.25 1.00 0.99 0.99 0.15 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.27 0.96 0.99 0.99 0.42 0.01 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.10 0.95 0.99 0.99 0.61 0.01 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.05 0.66 0.99 0.99 0.84 0.11 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.30 0.99 0.99 0.99 0.52 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.16
		// 0.75 0.99 0.99 0.99 0.87 0.04 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.04 0.77
		// 0.99 0.99 0.98 0.83 0.31 0.02 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.32 0.99
		// 0.99 0.98 0.56 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.18 0.90 0.99
		// 0.99 0.58 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.15 0.88 0.99
		// 0.91 0.17 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
		// 0.00 0.00 0.00 0.00";
		String[] datas = testDta.split("  ");

		double[] data = new double[datas.length];

		for (int i = 0; i < datas.length; i++)
			data[i] = Double.parseDouble(datas[i]);
		double[][] test = new double[1][0];
		test[0] = data;
		INDArray testArray = Nd4j.create(test);
		System.out.println(testArray);
		INDArray result1 = model.output(testArray);
		System.out.println(result1);
		int[] intArray = new int[10];
		String str = result1.toString();
		System.out.println(str);
		String value34 = result1.getRow(0).getColumn(0).toString();
		System.out.println(value34);
		str = str.replace("[", "").replace("]", "");
		String[] arr = str.split(",  ");
		System.out.println(arr[0] + "---" + arr[1]);
		for (int i = 0; i < 10; i++) {
			float f = Float.valueOf(arr[i]);
			System.out.println(f);
			int f2 = Math.round(f);
			// intArray[i]=t;
			intArray[i] = f2;
		}
		/*
		 * float[] f1 = new float[10]; int count2 = 0; for(int in : intArray) {
		 * f1[count2] = Float.valueOf(String.valueOf(in)); } INDArray ind =
		 * Nd4j.create(f1); System.out.println(ind);
		 */

		log.info("****************Example finished********************");
	}

	private static DataSetIterator getTrainingData() {
		float[][] featureData = new float[800][0];
		float[][] labelData = new float[800][0];

		for (int i = 0; i < 50; i++) {
			float[] features1 = buildFeatureVec("110000100100");
			featureData[i] = features1;
		}
		for (int i = 50; i < 100; i++) {
			float[] features2 = buildFeatureVec("110001100100");
			featureData[i] = features2;
		}
		for (int i = 100; i < 150; i++) {
			float[] features3 = buildFeatureVec("110010100100");
			featureData[i] = features3;
		}
		for (int i = 150; i < 200; i++) {
			float[] features4 = buildFeatureVec("110011100100");
			featureData[i] = features4;
		}
		for (int i = 200; i < 250; i++) {
			float[] features5 = buildFeatureVec("110100100100");
			featureData[i] = features5;
		}
		for (int i = 250; i < 300; i++) {
			float[] features6 = buildFeatureVec("110101100100");
			featureData[i] = features6;
		}
		for (int i = 300; i < 350; i++) {
			float[] features7 = buildFeatureVec("110110100100");
			featureData[i] = features7;
		}
		for (int i = 350; i < 400; i++) {
			float[] features8 = buildFeatureVec("110111100100");
			featureData[i] = features8;
		}
		for (int i = 400; i < 450; i++) {
			float[] features9 = buildFeatureVec("111000100100");
			featureData[i] = features9;
		}
		for (int i = 450; i < 500; i++) {
			float[] features10 = buildFeatureVec("111001100100");
			featureData[i] = features10;
		}
		for (int i = 500; i < 550; i++) {
			float[] features11 = buildFeatureVec("111010100100");
			featureData[i] = features11;
		}
		for (int i = 550; i < 600; i++) {
			float[] features12 = buildFeatureVec("111011100100");
			featureData[i] = features12;
		}
		for (int i = 600; i < 650; i++) {
			float[] features13 = buildFeatureVec("000001100100");
			featureData[i] = features13;
		}
		for (int i = 650; i < 700; i++) {
			float[] features14 = buildFeatureVec("000010100100");
			featureData[i] = features14;
		}
		for (int i = 700; i < 750; i++) {
			float[] features15 = buildFeatureVec("000011100100");
			featureData[i] = features15;
		}
		for (int i = 750; i < 800; i++) {
			float[] features16 = buildFeatureVec("000100100100");
			featureData[i] = features16;
		}

		/*
		 * featureData[0] = features1; featureData[1] = features2;
		 * featureData[2] = features3; featureData[3] = features4;
		 * featureData[4] = features5; featureData[5] = features6;
		 * featureData[6] = features7; featureData[7] = features8;
		 * featureData[8] = features9; featureData[9] = features10;
		 * featureData[10] = features11; featureData[11] = features12;
		 * featureData[12] = features13; featureData[13] = features14;
		 * featureData[14] = features15; featureData[15] = features16;
		 */
		for (int i = 0; i < 800;) {
			if (i == 0 || i < 50) {
				labelData[i++] = buildLabelVec("110000", "100100");
			} else if (i >= 50 && i < 100) {
				labelData[i++] = buildLabelVec("110001", "100100");
			} else if (i >= 100 && i < 150) {
				labelData[i++] = buildLabelVec("110010", "100100");
			} else if (i >= 150 && i < 200) {
				labelData[i++] = buildLabelVec("110011", "100100");
			} else if (i >= 200 && i < 250) {
				labelData[i++] = buildLabelVec("110100", "100100");
			} else if (i >= 250 && i < 300) {
				labelData[i++] = buildLabelVec("110101", "100100");
			} else if (i >= 300 && i < 350) {
				labelData[i++] = buildLabelVec("110110", "100100");
			} else if (i >= 350 && i < 400) {
				labelData[i++] = buildLabelVec("110111", "100100");
			} else if (i >= 400 && i < 450) {
				labelData[i++] = buildLabelVec("111000", "100100");
			} else if (i >= 450 && i < 500) {
				labelData[i++] = buildLabelVec("111001", "100100");
			} else if (i >= 500 && i < 550) {
				labelData[i++] = buildLabelVec("111010", "100100");
			} else if (i >= 550 && i < 600) {
				labelData[i++] = buildLabelVec("111011", "100100");
			} else if (i >= 600 && i < 650) {
				labelData[i++] = buildLabelVec("000001", "100100");
			} else if (i >= 650 && i < 700) {
				labelData[i++] = buildLabelVec("000010", "100100");
			} else if (i >= 700 && i < 750) {
				labelData[i++] = buildLabelVec("000011", "100100");
			} else if (i >= 750 && i < 800) {
				labelData[i++] = buildLabelVec("000100", "100100");
			}

		}

		/*
		 * labelData[0] = label1; labelData[1] = label2; labelData[2] = label3;
		 * labelData[3] = label4; labelData[4] = label5; labelData[5] = label6;
		 * labelData[6] = label7; labelData[7] = label8; labelData[8] = label9;
		 * labelData[9] = label10; labelData[10] = label11; labelData[11] =
		 * label12; labelData[12] = label13; labelData[13] = label14;
		 * labelData[14] = label15; labelData[15] = label16;
		 */

		INDArray features = Nd4j.create(featureData);
		INDArray labels = Nd4j.create(labelData);
		DataSet dataSet = new DataSet(features, labels);

		List<DataSet> listDataSets = dataSet.asList();
		Collections.shuffle(listDataSets, rng);
		DataSetIterator dsItr = new ListDataSetIterator(listDataSets, 8);
		return dsItr;
	}

	private static float[] buildFeatureVec(String text) {
		float[] featureVec = new float[28 * 28];
		int textLength = text.length();
		for (int i = 0; i < textLength; i++) {
			float f = Float.valueOf(text.substring(i, i + 1));
			featureVec[i] = f;
		}

		for (int i = textLength; i < 784; i++) {
			featureVec[i] = 0.0F;
		}

		return featureVec;
	}

	private static float[] buildLabelVec(String text, String label) {
		float[] labelVec = new float[6];
		int textLength = text.length();
		for (int i = 0; i < textLength; i++) {
			int inputBit = Integer.valueOf(text.substring(i, i + 1));
			int labelBit = Integer.valueOf(label.substring(i, i + 1));
			int temp = inputBit ^ labelBit;
			float f = Float.valueOf(String.valueOf(temp));
			labelVec[i] = f;
		}

		return labelVec;
	}

	/*
	 * public static DataSetIterator load(String filePath, String tag) {
	 * double[][] inputs = new double[832][784]; double[][] labels = new
	 * double[832][10]; File file = new File(filePath); DataSetIterator dsItr =
	 * null; try {
	 * 
	 * BufferedReader in = new BufferedReader(new FileReader(file)); String
	 * line; int count = 0; while ((line = in.readLine()) != null) { String[]
	 * datas = line.split(tag); if (datas.length == 0) continue; double[] data =
	 * new double[datas.length - 1]; double[] label = new double[10]; for (int i
	 * = 0; i < datas.length - 1; i++) data[i] = Double.parseDouble(datas[i]);
	 * int labelValue = Integer.valueOf(datas[datas.length - 1]); label =
	 * getLabelValue(labelValue); inputs[count] = data; labels[count] = label; }
	 * in.close();
	 * 
	 * INDArray featuresArray = Nd4j.create(inputs); INDArray labelsArray =
	 * Nd4j.create(labels); DataSet dataSet = new DataSet(featuresArray,
	 * labelsArray);
	 * 
	 * List<DataSet> listDataSets = dataSet.asList(); //
	 * Collections.shuffle(listDataSets, rng); dsItr = new
	 * ListDataSetIterator(listDataSets, 60);
	 * 
	 * } catch (IOException e) { e.printStackTrace(); return null; } //
	 * System.out.println("导入数据:" + dataset.size()); return dsItr; }
	 */

	public static DataSetIterator load(String filePath, String tag) {
		double[][] inputs = new double[24012][784];
		double[][] labels = new double[24012][10];
		File file = new File(filePath);
		DataSetIterator dsItr = null;
		try {

			BufferedReader in = new BufferedReader(new FileReader(file));
			String line;
			int count = 0;
			while ((line = in.readLine()) != null) {
				String[] datas = line.split(tag);
				if (datas.length == 0)
					continue;
				double[] data = new double[datas.length];

				for (int i = 0; i < datas.length; i++)
					data[i] = Double.parseDouble(datas[i]);

				inputs[count++] = data;

			}
			in.close();

			BufferedReader in1 = new BufferedReader(new FileReader("D:/Backup/TEMP/Keystore/GeneratedLabel.txt"));
			String line1;
			int count1 = 0;
			while ((line1 = in1.readLine()) != null) {
				String[] datas = line1.split(tag);
				if (datas.length == 0)
					continue;
				double[] data = new double[datas.length];

				for (int i = 0; i < datas.length; i++)
					data[i] = Double.parseDouble(datas[i]);

				labels[count1++] = data;

			}
			in1.close();

			INDArray featuresArray = Nd4j.create(inputs);
			INDArray labelsArray = Nd4j.create(labels);
			DataSet dataSet = new DataSet(featuresArray, labelsArray);

			List<DataSet> listDataSets = dataSet.asList();
			// Collections.shuffle(listDataSets, rng);
			dsItr = new ListDataSetIterator(listDataSets, 60);

		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		// System.out.println("导入数据:" + dataset.size());
		return dsItr;
	}

	public static void write(INDArray in, PrintWriter writer) {

		try {

			writer.write(in.toString() + "\n");

		} catch (Exception e) {
			e.printStackTrace();

		}
		// System.out.println("导入数据:" + dataset.size());

	}

	private static double[] getLabelValue(int value) {
		double[] label = new double[10];
		if (value == 0) {
			label = new double[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		} else if (value == 1) {
			label = new double[] { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		} else if (value == 2) {
			label = new double[] { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		} else if (value == 3) {
			label = new double[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		} else if (value == 4) {
			label = new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		} else if (value == 5) {
			label = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
		} else if (value == 6) {
			label = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
		} else if (value == 7) {
			label = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };
		} else if (value == 8) {
			label = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
		} else if (value == 9) {
			label = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
		}

		return label;
	}

}
