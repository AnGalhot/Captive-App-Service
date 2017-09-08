package com.fis.neural.key.synchronize;

public interface GlobalConstants {
	public static final int inputLayerNeurons = 10;
	public static final int hiddenLayerNeurons = 10;
	public static final int outputLayerNeurons = 1;
	public static final int iterations = 1;
	public static final double learningRate = 0.01;
	public static final double momentum = 0.9;
	public static final int nEpochs = 1;
	public static final String WEIGHT_MATRIX_KEY = "W";
	public static final String HIDDEN_LAYER_NAME = "hiddenLayer";
	public static final String OUTPUT_LAYER_NAME = "OutputLayer";
	public static final int MIN_RANGE = 0;
	public static final int MAX_RANGE = 3;
	public static final int nExamples = 20;
	public static final int MIN_WEIGHT_RANGE = -4;
	public static final int MAX_WEIGHT_RANGE = 4;

}
