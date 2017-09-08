package com.fis.adversarial.receiver.cryptography;

import static com.fis.adversarial.cryptography.GlobalConstants.FC_OUTPUT_LAYER;
import static com.fis.adversarial.cryptography.GlobalConstants.FIRST_CONVOLUTION_LAYER;
import static com.fis.adversarial.cryptography.GlobalConstants.FIRST_CONVOLUTION_OUT_DEPTH;
import static com.fis.adversarial.cryptography.GlobalConstants.FIRST_POOLING_LAYER;
import static com.fis.adversarial.cryptography.GlobalConstants.HIDDEN_LAYER;
import static com.fis.adversarial.cryptography.GlobalConstants.HIDDEN_LAYER_OUT_DEPTH;
import static com.fis.adversarial.cryptography.GlobalConstants.L2;
import static com.fis.adversarial.cryptography.GlobalConstants.SECOND_CONVOLUTION_LAYER;
import static com.fis.adversarial.cryptography.GlobalConstants.SECOND_CONVOLUTION_OUT_DEPTH;
import static com.fis.adversarial.cryptography.GlobalConstants.SECOND_POOLING_LAYER;
import static com.fis.adversarial.cryptography.GlobalConstants.convLayerkernelSize;
import static com.fis.adversarial.cryptography.GlobalConstants.depth;
import static com.fis.adversarial.cryptography.GlobalConstants.height;
import static com.fis.adversarial.cryptography.GlobalConstants.iterations;
import static com.fis.adversarial.cryptography.GlobalConstants.learningRate;
import static com.fis.adversarial.cryptography.GlobalConstants.nChannels;
import static com.fis.adversarial.cryptography.GlobalConstants.nEpochs;
import static com.fis.adversarial.cryptography.GlobalConstants.outputNum;
import static com.fis.adversarial.cryptography.GlobalConstants.poolLayerStride;
import static com.fis.adversarial.cryptography.GlobalConstants.poolLayerkernelSize;
import static com.fis.adversarial.cryptography.GlobalConstants.seed;
import static com.fis.adversarial.cryptography.GlobalConstants.stride;
import static com.fis.adversarial.cryptography.GlobalConstants.useRegularization;
import static com.fis.adversarial.cryptography.GlobalConstants.width;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import static com.fis.adversarial.cryptography.GlobalConstants.*;
import com.fis.adversarial.cryptography.AdversarialNeuralHelper;

public class ReceiverConvolutionNN {
	private static final Logger log = LoggerFactory.getLogger(ReceiverConvolutionNN.class);
	private static final Random rng = new Random(seed);
	private MultiLayerNetwork model;

	//private default no-arg constructor
	private ReceiverConvolutionNN() {

	}

	//factory method to create instance of this class
	public static ReceiverConvolutionNN getInstance() {
		return new ReceiverConvolutionNN();
	}

	//construct the deep learning convolutional neural network(CNN)
	private void buildNeuralNetworkModel() {
		MultiLayerNetwork model = constructConvolutionalNeuralNetwork();
		this.model = model;
	}

	private MultiLayerNetwork constructConvolutionalNeuralNetwork() {
		// create Network
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		// setup basic network configurations
		setNetworkConfigurations(builder);
		ListBuilder layersBuilder = builder.list();
		AdversarialNeuralHelper helper = AdversarialNeuralHelper.getInstance();
		/*
		 * start with convolution layer
		 * 
		 */
		ConvolutionLayer firstConvLayer = helper.convolutionLayer(convLayerkernelSize, stride, FIRST_CONVOLUTION_LAYER,
				nChannels, FIRST_CONVOLUTION_OUT_DEPTH);
		layersBuilder.layer(0, firstConvLayer);
		/*
		 * pooling layer
		 * 
		 */
		SubsamplingLayer firstPoolingLayer = helper.poolingLayer(FIRST_POOLING_LAYER, poolLayerkernelSize,
				poolLayerStride);
		layersBuilder.layer(1, firstPoolingLayer);
		/*
		 * convolution layer
		 */
		ConvolutionLayer secondConvLayer = helper.convolutionLayer(convLayerkernelSize, stride,
				SECOND_CONVOLUTION_LAYER, SECOND_CONVOLUTION_OUT_DEPTH);
		layersBuilder.layer(2, secondConvLayer);
		/*
		 * pooling layer
		 * 
		 */
		SubsamplingLayer secondPoolingLayer = helper.poolingLayer(SECOND_POOLING_LAYER, poolLayerkernelSize,
				poolLayerStride);
		layersBuilder.layer(3, secondPoolingLayer);
		/*
		 * hidden layer
		 */
		DenseLayer hiddenLayer = helper.hiddenLayer(HIDDEN_LAYER, HIDDEN_LAYER_OUT_DEPTH);
		layersBuilder.layer(4, hiddenLayer);
		/*
		 * fully connected output layer
		 */
		OutputLayer outputLayer = helper.fullyConnectedOutputLayer(FC_OUTPUT_LAYER, nReceiverNNOutcome);
		layersBuilder.layer(5, outputLayer);
		// input is a "flattened" row vector format (i.e., 1x784 vectors),hence
		// the "convolutionalFlat" input type used here.
		layersBuilder.setInputType(InputType.convolutionalFlat(height, width, depth));
		layersBuilder.pretrain(false);
		layersBuilder.backprop(true);

		MultiLayerConfiguration conf = layersBuilder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(1));

		return model;

	}

	private void setNetworkConfigurations(NeuralNetConfiguration.Builder builder) {
		builder.iterations(iterations);
		// learning rate
		builder.learningRate(learningRate);
		builder.seed(seed);
		builder.regularization(useRegularization);
		builder.l2(L2);
		builder.learningRateDecayPolicy(LearningRatePolicy.Schedule);
		builder.learningRateSchedule(getLearningRateSchedule());
		builder.weightInit(WeightInit.XAVIER);
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		builder.updater(Updater.NESTEROVS);

	}

	// do training the model
	public void doTrainModel() throws Exception {
		DataSetIterator dataSetItr = AdversarialNeuralHelper.getInstance().loadTrainingData(RECEIVER);
		for (int i = 0; i < nEpochs; i++) {
			model.fit(dataSetItr);
		}
		saveTrainedConvolutionalModel(model);
	}

	// Save a trained neural network model
	public void saveTrainedConvolutionalModel(MultiLayerNetwork model) throws Exception {
		AdversarialNeuralHelper.getInstance().saveTrainedModel(model,RECEIVER);
	}

	public INDArray evaluateResult(INDArray input) throws Exception {
		if (this.model == null) {
			this.model = AdversarialNeuralHelper.getInstance().loadTrainedModel(RECEIVER);
		}
		INDArray outcome = this.model.output(input);
		return outcome;
	}

	private Map<Integer, Double> getLearningRateSchedule() {
		// learning rate schedule in the form of <Iteration #, Learning Rate>
		Map<Integer, Double> lrSchedule = new HashMap<>();
		lrSchedule.put(0, 0.01);
		lrSchedule.put(1000, 0.005);
		lrSchedule.put(3000, 0.001);
		return lrSchedule;
	}

	public MultiLayerNetwork getModel() {
		return model;
	}

	public static void main(String[] args) throws Exception {
		ReceiverConvolutionNN convolutionNN = new ReceiverConvolutionNN();
		convolutionNN.buildNeuralNetworkModel();
		convolutionNN.doTrainModel();
	}

}
