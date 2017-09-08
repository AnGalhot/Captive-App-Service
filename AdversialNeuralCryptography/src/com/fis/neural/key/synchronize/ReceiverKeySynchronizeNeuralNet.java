package com.fis.neural.key.synchronize;

import static com.fis.neural.key.synchronize.GlobalConstants.WEIGHT_MATRIX_KEY;
import static com.fis.neural.key.synchronize.GlobalConstants.hiddenLayerNeurons;
import static com.fis.neural.key.synchronize.GlobalConstants.inputLayerNeurons;
import static com.fis.neural.key.synchronize.GlobalConstants.iterations;
import static com.fis.neural.key.synchronize.GlobalConstants.learningRate;
import static com.fis.neural.key.synchronize.GlobalConstants.momentum;
import static com.fis.neural.key.synchronize.GlobalConstants.nEpochs;
import static com.fis.neural.key.synchronize.GlobalConstants.outputLayerNeurons;
import static com.fis.neural.key.synchronize.GlobalConstants.HIDDEN_LAYER_NAME;
import static com.fis.neural.key.synchronize.GlobalConstants.OUTPUT_LAYER_NAME;
import static com.fis.neural.key.synchronize.GlobalConstants.MIN_WEIGHT_RANGE;
import static com.fis.neural.key.synchronize.GlobalConstants.MAX_WEIGHT_RANGE;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ReceiverKeySynchronizeNeuralNet {

	private static final int seed = 54321;


	private static final Random rng = new Random(seed);
	private MultiLayerNetwork model;
	

	private ReceiverKeySynchronizeNeuralNet() {
		buildNeuralNetworkModel();
	}

	public static ReceiverKeySynchronizeNeuralNet getInstance() {
		return new ReceiverKeySynchronizeNeuralNet();
	}

	private void buildNeuralNetworkModel() {
		MultiLayerNetwork model = constructNeuralNetwork();
		this.model = model;
	}

	private MultiLayerNetwork constructNeuralNetwork() {

		// create Network
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();

		// setup basic network confiurations
		setNetworkConfigurations(builder);

		ListBuilder layersBuilder = builder.list();
		// start with hidden layer as network will create input layer implicitly
		// with number of neurons based on inputLayerNeurons parameter specified
		// here
		DenseLayer hiddenLayer = buildHiddenLayer();
		layersBuilder.layer(0, hiddenLayer);
		// output layer
		OutputLayer outputLayer = buildOutputLayer();
		layersBuilder.layer(1, outputLayer);

		layersBuilder.pretrain(false);
		layersBuilder.backprop(true);

		MultiLayerConfiguration conf = layersBuilder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(1));

		return model;

	}

	private DenseLayer buildHiddenLayer() {
		DenseLayer.Builder inputLayerBuilder = new DenseLayer.Builder();
		inputLayerBuilder.name(HIDDEN_LAYER_NAME);
		inputLayerBuilder.nIn(inputLayerNeurons);
		inputLayerBuilder.nOut(hiddenLayerNeurons);
		inputLayerBuilder.activation(Activation.RELU);		
		inputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		inputLayerBuilder.dist(new UniformDistribution(MIN_WEIGHT_RANGE, MAX_WEIGHT_RANGE));
		 
		return inputLayerBuilder.build();
	}

	private OutputLayer buildOutputLayer() {
		OutputLayer.Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.MSE);
		outputLayerBuilder.name(OUTPUT_LAYER_NAME);
		outputLayerBuilder.nIn(hiddenLayerNeurons);
		outputLayerBuilder.nOut(outputLayerNeurons);
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		outputLayerBuilder.dist(new UniformDistribution(MIN_WEIGHT_RANGE, MAX_WEIGHT_RANGE));
		return outputLayerBuilder.build();
	}

	private void setNetworkConfigurations(NeuralNetConfiguration.Builder builder) {
		builder.iterations(iterations);
		// learning rate
		builder.learningRate(learningRate);
		builder.seed(rng.nextLong());
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.weightInit(WeightInit.XAVIER);
		builder.updater(Updater.NESTEROVS);
		builder.momentum(momentum);

	}

	public void trainModel(DataSet trainData) {
		this.model.fit(trainData);
	}

	public double computeOutput(INDArray input, INDArray label) {
		this.model.setInput(input);
		this.model.setLabels(label);
		// this.model.computeGradientAndScore();
		for (int i = 0; i < nEpochs; i++) {
			this.model.fit();
		}
		double score = this.model.score();
		
		return score;
	}
	
	public INDArray computeWeight(INDArray input, INDArray label) {
		this.model.setInput(input);
		this.model.setLabels(label);
		// this.model.computeGradientAndScore();
		for (int i = 0; i < nEpochs; i++) {
			this.model.fit();
		}
		INDArray result = this.model.getLayer(1).getParam(WEIGHT_MATRIX_KEY);
		return result;
	}

	public INDArray evaluateModel(MultiLayerNetwork multiLayerNetwork, DataSet testDataSet) {
		// Evaluation eval = new Evaluation(1); //create an evaluation object
		// with 1 possible outcome
		INDArray outcome = multiLayerNetwork.output(testDataSet.getFeatureMatrix());
		return outcome;
	}

	public void applyLearningRule(DataSet dataSet) {
		this.trainModel(dataSet);
	}

	public INDArray getWeightMatrix() {
		INDArray weights = this.model.getLayer(0).getParam(WEIGHT_MATRIX_KEY);
		return weights;
	}
	
	public MultiLayerNetwork getModel() {
		return this.model;
	}

}
