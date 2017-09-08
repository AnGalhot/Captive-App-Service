package com.fis.adversarial.cryptography;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import static com.fis.adversarial.cryptography.GlobalConstants.*;

public class AdversarialNeuralHelper {

	private static final Logger log = LoggerFactory.getLogger(AdversarialNeuralHelper.class);

	private static final boolean saveUpdater = true;

	// default private constructor
	private AdversarialNeuralHelper() {

	}

	/*
	 * this method added to get the instance of this class. Note : this is only
	 * way to create a instance of this class
	 */
	public static AdversarialNeuralHelper getInstance() {
		return new AdversarialNeuralHelper();
	}

	/**
	 * it loads the training and corresponding label data and wrap it in
	 * DataSetIterator instance
	 * 
	 * @return dataSetIterator
	 * @throws Exception
	 */
	public DataSetIterator loadTrainingData(String networkType) throws Exception {
		if (CONVEYOR.equals(networkType)) {
			return loadDataForConveyorNN();
		} else if (RECEIVER.equals(networkType)) {
			return loadDataForReceiverNN();
		} else if (EAVESDROPPER.equals(networkType)) {
			return loadDataForEavesdropperNN();
		} else {
			throw new IllegalArgumentException("Invalid Network type passed ; Please provide valid argument !!!");
		}
	}

	private DataSetIterator loadDataForConveyorNN() throws Exception {
		double[][] inputs = new double[nExamples][rows * columns];
		double[][] labels = new double[nLabels][nOutcome];
		this.buildTrainingData(inputs, conveyorTrainingData);
		this.buildLabelData(labels, conveyorLabelData);
		DataSetIterator dataSetIterator = this.createDataSetIterator(inputs, labels);
		return dataSetIterator;
	}

	private DataSetIterator loadDataForReceiverNN() throws Exception {
		double[][] inputs = new double[nExamples][rows * columns];
		double[][] labels = new double[nLabels][nReceiverNNOutcome];
		this.buildTrainingData(inputs, receiverTrainingData);
		this.buildLabelData(labels, receiverLabelData);
		DataSetIterator dataSetIterator = this.createDataSetIterator(inputs, labels);
		return dataSetIterator;
	}

	private DataSetIterator loadDataForEavesdropperNN() throws Exception {
		double[][] inputs = new double[nExamples][rows * columns];
		double[][] labels = new double[nLabels][nEavesdropperNNOutcome];
		this.buildTrainingData(inputs, eavesdropperTrainingData);
		this.buildLabelData(labels, eavesdropperLabelData);
		DataSetIterator dataSetIterator = this.createDataSetIterator(inputs, labels);
		return dataSetIterator;
	}

	private void buildTrainingData(double[][] inputs, String trainingData) throws Exception {
		BufferedReader in = null;
		try {
			in = new BufferedReader(new FileReader(new File(trainingData)));
			String line;
			int count = 0;
			while ((line = in.readLine()) != null) {
				String[] datas = line.split(delimiter);
				if (datas.length == 0)
					continue;
				double[] data = new double[datas.length];

				for (int i = 0; i < datas.length; i++)
					data[i] = Double.parseDouble(datas[i]);

				inputs[count++] = data;

			}
		} catch (Exception e) {
			log.error("Exception occurred while loading training data...", e);
			throw e;
		} finally {
			if (in != null) {
				in.close();
			}
		}
	}

	private void buildLabelData(double[][] labels, String labelData) throws Exception {

		BufferedReader in = null;
		try {
			in = new BufferedReader(new FileReader(new File(labelData)));
			String line;
			int count = 0;
			while ((line = in.readLine()) != null) {
				String[] datas = line.split(delimiter);
				if (datas.length == 0)
					continue;
				double[] data = new double[datas.length];

				for (int i = 0; i < datas.length; i++)
					data[i] = Double.parseDouble(datas[i]);

				labels[count++] = data;

			}
		} catch (Exception e) {
			log.error("Exception occurred while loading training data...", e);
			throw e;
		} finally {
			if (in != null) {
				in.close();
			}
		}

	}

	private DataSetIterator createDataSetIterator(double[][] inputs, double[][] labels) {
		INDArray inputsArray = Nd4j.create(inputs);
		INDArray labelsArray = Nd4j.create(labels);
		DataSet dataSet = new DataSet(inputsArray, labelsArray);
		List<DataSet> listDataSets = dataSet.asList();
		DataSetIterator dsItr = new ListDataSetIterator<DataSet>(listDataSets, batchSize);
		return dsItr;
	}

	public OutputLayer fullyConnectedOutputLayer(String layerName, int nOut) {
		OutputLayer.Builder builder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
		builder.nOut(nOut);
		builder.name(layerName);
		builder.activation(Activation.SOFTMAX);
		OutputLayer outputLayer = builder.build();
		return outputLayer;
	}

	public DenseLayer hiddenLayer(String layerName, int nOut) {
		DenseLayer.Builder builder = new DenseLayer.Builder();
		builder.activation(Activation.RELU);
		builder.nOut(nOut);
		builder.name(layerName);
		DenseLayer hiddenLayer = builder.build();
		return hiddenLayer;
	}

	// build convolution layer based on various parameters
	public ConvolutionLayer convolutionLayer(int[] kernelSize, int[] stride, String layerName, int inputDepth,
			int outputDepth) {
		ConvolutionLayer.Builder convLayerBuilder = new ConvolutionLayer.Builder(kernelSize);
		convLayerBuilder.nIn(inputDepth); // nIn here is the nChannels
		convLayerBuilder.nOut(outputDepth); // nOut is the number of filters to
											// be applied
		convLayerBuilder.stride(stride);
		convLayerBuilder.activation(Activation.IDENTITY);
		convLayerBuilder.name(layerName);
		ConvolutionLayer convolutionLayer = convLayerBuilder.build();
		return convolutionLayer;
	}

	public ConvolutionLayer convolutionLayer(int[] kernelSize, int[] stride, String layerName, int outputDepth) {
		ConvolutionLayer.Builder convLayerBuilder = new ConvolutionLayer.Builder(kernelSize);
		convLayerBuilder.nOut(outputDepth); // nOut is the number of filters to
											// be applied
		convLayerBuilder.stride(stride);
		convLayerBuilder.activation(Activation.IDENTITY);
		convLayerBuilder.name(layerName);
		ConvolutionLayer convolutionLayer = convLayerBuilder.build();
		return convolutionLayer;
	}

	public SubsamplingLayer poolingLayer(String layerName, int[] kernelSize, int[] stride) {
		SubsamplingLayer.Builder builder = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX);
		builder.kernelSize(kernelSize);
		builder.stride(stride);
		builder.name(layerName);
		SubsamplingLayer subsamplingLayer = builder.build();
		return subsamplingLayer;
	}

	public void saveTrainedModel(MultiLayerNetwork model, String networkType) throws Exception {
		if (CONVEYOR.equals(networkType)) {
			saveConveyorTrainedModel(model);
		} else if (RECEIVER.equals(networkType)) {
			saveReceiverTrainedModel(model);
		} else if (EAVESDROPPER.equals(networkType)) {
			saveEavesdropperTrainedModel(model);
		} else {
			throw new IllegalArgumentException("Invalid Network type passed ; Please provide valid argument !!!");
		}
	}

	private void saveConveyorTrainedModel(MultiLayerNetwork model) throws Exception {
		try {
			ModelSerializer.writeModel(model, conveyorModelPath, saveUpdater);
		} catch (IOException e) {
			log.error("Exception occured while saving conveyor trained model...", e);
			throw e;
		}
	}

	private void saveReceiverTrainedModel(MultiLayerNetwork model) throws Exception {
		try {
			ModelSerializer.writeModel(model, receiverModelPath, saveUpdater);
		} catch (IOException e) {
			log.error("Exception occured while saving receiver trained model...", e);
			throw e;
		}
	}

	private void saveEavesdropperTrainedModel(MultiLayerNetwork model) throws Exception {
		try {
			ModelSerializer.writeModel(model, eavesdropperModelPath, saveUpdater);
		} catch (IOException e) {
			log.error("Exception occured while saving eavesdropper trained model...", e);
			throw e;
		}
	}

	public MultiLayerNetwork loadTrainedModel(String networkType) throws Exception {
		if (CONVEYOR.equals(networkType)) {
			return loadConveyorTrainedModel();
		} else if (RECEIVER.equals(networkType)) {
			return loadReceiverTrainedModel();
		} else if (EAVESDROPPER.equals(networkType)) {
			return loadEavesdropperTrainedModel();
		} else {
			throw new IllegalArgumentException("Invalid Network type passed ; Please provide valid argument !!!");
		}
	}

	private MultiLayerNetwork loadConveyorTrainedModel() throws Exception {
		try {
			MultiLayerNetwork restoredModel = ModelSerializer.restoreMultiLayerNetwork(conveyorModelPath);
			return restoredModel;
		} catch (IOException e) {
			log.error("Exception occured while loading conveyor trained model...", e);
			throw e;
		}
	}

	private MultiLayerNetwork loadReceiverTrainedModel() throws Exception {
		try {
			MultiLayerNetwork restoredModel = ModelSerializer.restoreMultiLayerNetwork(receiverModelPath);
			return restoredModel;
		} catch (IOException e) {
			log.error("Exception occured while loading receiver trained model...", e);
			throw e;
		}
	}

	private MultiLayerNetwork loadEavesdropperTrainedModel() throws Exception {
		try {
			MultiLayerNetwork restoredModel = ModelSerializer.restoreMultiLayerNetwork(eavesdropperModelPath);
			return restoredModel;
		} catch (IOException e) {
			log.error("Exception occured while loading eavesdropper trained model...", e);
			throw e;
		}
	}

	public INDArray doCharacterNormalization(char ch) {
		ConfigPropertiesFileHandler handler = ConfigPropertiesFileHandler.getInstance();
		INDArray normalizedValue = handler.getNormalizedValue(String.valueOf(ch));
		return normalizedValue;
	}

	public INDArray doEncryptedTokenNormalization(String token) {
		com.fis.adversarial.receiver.cryptography.ConfigPropertiesFileHandler handler = com.fis.adversarial.receiver.cryptography.ConfigPropertiesFileHandler
				.getInstance();
		INDArray normalizedValue = handler.getNormalizedValue(token);
		return normalizedValue;
	}
	
	public INDArray doTokenNormalizationByEavesdropper(String token) {
		com.fis.adversarial.receiver.cryptography.ConfigPropertiesFileHandler handler = com.fis.adversarial.receiver.cryptography.ConfigPropertiesFileHandler
				.getInstance();
		INDArray normalizedValue = handler.getNormalizedValue(token);
		return normalizedValue;
	}

}
