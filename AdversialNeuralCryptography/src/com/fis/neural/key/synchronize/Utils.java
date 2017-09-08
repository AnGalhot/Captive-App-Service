package com.fis.neural.key.synchronize;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static com.fis.neural.key.synchronize.GlobalConstants.*;

public class Utils {

	private static INDArray[] cacheWeights = null;

	public static Object[] generateRandomInputVector(Random random) {

		double[] sum = new double[nExamples];
		double[] input1 = new double[nExamples];
		double[] input2 = new double[nExamples];
		double[] input3 = new double[nExamples];
		double[] input4 = new double[nExamples];

		double[] input5 = new double[nExamples];
		double[] input6 = new double[nExamples];
		double[] input7 = new double[nExamples];
		double[] input8 = new double[nExamples];
		double[] input9 = new double[nExamples];
		double[] input10 = new double[nExamples];

		for (int i = 0; i < nExamples; i++) {
			input1[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
			input2[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
			input3[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
			input4[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();

			input5[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
			input6[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
			input7[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
			input8[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
			input9[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
			input10[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();

			sum[i] = input1[i] + input2[i] + input3[i] + input4[i] + input5[i] + input6[i] + input7[i] + input8[i]
					+ input9[i] + input10[i];

			sum[i] = input1[i] + input2[i] + input3[i] + input4[i];
		}
		INDArray inputNDArray1 = Nd4j.create(input1, new int[] { nExamples, 1 });
		INDArray inputNDArray2 = Nd4j.create(input2, new int[] { nExamples, 1 });
		INDArray inputNDArray3 = Nd4j.create(input2, new int[] { nExamples, 1 });
		INDArray inputNDArray4 = Nd4j.create(input2, new int[] { nExamples, 1 });

		INDArray inputNDArray5 = Nd4j.create(input2, new int[] { nExamples, 1 });
		INDArray inputNDArray6 = Nd4j.create(input2, new int[] { nExamples, 1 });
		INDArray inputNDArray7 = Nd4j.create(input2, new int[] { nExamples, 1 });
		INDArray inputNDArray8 = Nd4j.create(input2, new int[] { nExamples, 1 });
		INDArray inputNDArray9 = Nd4j.create(input2, new int[] { nExamples, 1 });
		INDArray inputNDArray10 = Nd4j.create(input2, new int[] { nExamples, 1 });
		INDArray inputNDArray = Nd4j.hstack(inputNDArray1, inputNDArray2, inputNDArray3, inputNDArray4, inputNDArray5,
				inputNDArray6, inputNDArray7, inputNDArray8, inputNDArray9, inputNDArray10);
		//INDArray inputNDArray = Nd4j.hstack(inputNDArray1, inputNDArray2, inputNDArray3, inputNDArray4);
		//INDArray inputNDArray = Nd4j.hstack(inputNDArray1, inputNDArray2, inputNDArray3, inputNDArray4);
		INDArray output = Nd4j.create(sum, new int[] { nExamples, 1 });
		DataSetIterator dsItr = buildDataSetItr(inputNDArray, output, random);
		return new Object[] { inputNDArray, output, dsItr };
	}

	public static DataSetIterator buildDataSetItr(INDArray input, INDArray output, Random random) {
		DataSet dataSet = new DataSet(input, output);
		List<DataSet> dataSets = dataSet.asList();
		Collections.shuffle(dataSets, random);
		return new ListDataSetIterator(dataSets);
	}

	public static boolean areSameOutput(double score1, double score2) {
		if (score1 == score2) {
			return true;
		}
		return false;
	}

	public static DenseLayer buildHiddenLayer() {
		DenseLayer.Builder inputLayerBuilder = new DenseLayer.Builder();
		inputLayerBuilder.name(HIDDEN_LAYER_NAME);
		inputLayerBuilder.nIn(inputLayerNeurons);
		inputLayerBuilder.nOut(hiddenLayerNeurons);
		inputLayerBuilder.activation(Activation.RELU);
		inputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		inputLayerBuilder.dist(new UniformDistribution(MIN_WEIGHT_RANGE, MAX_WEIGHT_RANGE));
		return inputLayerBuilder.build();
	}

	public static OutputLayer buildOutputLayer() {
		OutputLayer.Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.MSE);
		outputLayerBuilder.name(OUTPUT_LAYER_NAME);
		outputLayerBuilder.nIn(hiddenLayerNeurons);
		outputLayerBuilder.nOut(outputLayerNeurons);
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		outputLayerBuilder.dist(new UniformDistribution(MIN_WEIGHT_RANGE, MAX_WEIGHT_RANGE));
		return outputLayerBuilder.build();
	}

	public static void updateWeightsUsingHebbianLearningRule(MultiLayerNetwork model, double conveyorOutput,
			double receiverOutput) {
		INDArray weightsMatrix = model.getLayer(0).getParam(WEIGHT_MATRIX_KEY);
		INDArray inputVector = model.getInput();
		int nWeightsMatrixRows = weightsMatrix.rows();
		INDArray inputArray = inputVector.getRow(0);
		INDArray[] wij = new INDArray[nWeightsMatrixRows];
		for (int i = 0; i < nWeightsMatrixRows; i++) {
			INDArray weightRow = weightsMatrix.getRow(i);
			INDArray hiddenUnitOutput = calculateOutputOfHiddenUnit(inputArray, weightRow);
			double sum = hiddenUnitOutput.sumNumber().doubleValue();
			int signumValue = sigma(sum);
			INDArray intermediateRes = (inputArray.mul(signumValue))
					.mul(theta(signumValue * conveyorOutput) * (theta(conveyorOutput * receiverOutput)));
			INDArray weightsAdjustment = weightRow.add(intermediateRes);
			// System.out.println(weightsAdjustment);
			fitWeightInRange(weightsAdjustment, i);
			wij[i] = weightsAdjustment;
		}
		cacheWeights = wij;
		INDArray result = Nd4j.hstack(wij);
		// System.out.println(result);
		model.getLayer(0).setParam(WEIGHT_MATRIX_KEY, result);
	}

	private static int theta(double r) {
		return (r >= 0) ? 1 : 0;
	}

	private static void fitWeightInRange(INDArray input, int currentIndex) {
		int count = input.columns();
		for (int i = 0; i < count; i++) {
			double currentWeight = Double.valueOf(input.getColumn(i).toString());
			INDArray temp = null;
			if (cacheWeights != null) {
				temp = cacheWeights[currentIndex];
				double tempWeight = Double.valueOf(temp.getColumn(i).toString());
				if (tempWeight == 0.0) {
					input.putScalar(i, 0);
				} else {
					input.putScalar(i, 1);
				}
			} else if ((currentWeight) > 2.0) {
				input.putScalar(i, 1);
			} else {
				input.putScalar(i, 0);
			}
		}
	}

	public static int sigma(double r) {
		return (r > 0) ? 1 : -1;
	}

	private static INDArray calculateOutputOfHiddenUnit(INDArray input, INDArray weights) {

		INDArray tempWeights = weights;
		INDArray tempInput = input;
		int columns = tempInput.columns();
		double[] wixj = new double[columns];
		for (int i = 0; i < columns; i++) {
			String str1 = tempInput.getColumn(i).toString();
			wixj[i] = (Double.valueOf(tempWeights.getColumn(i).toString())) * (Double.valueOf(str1));
		}
		INDArray result = Nd4j.create(wixj);
		return result;
	}

	public static String generateKeyFromWeights(MultiLayerNetwork model) {
		INDArray syncWeightsMatrix = model.getLayer(0).getParam(WEIGHT_MATRIX_KEY);
		int nRows = syncWeightsMatrix.rows();
		// StringBuilder builder = new StringBuilder();
		String buffer = null;
		INDArray row = null;
		for (int i = 0; i < nRows; i++) {
			if (row != null) {
				Object[] objs = performXOR(row, syncWeightsMatrix.getRow(i));
				row = (INDArray) objs[0];
				buffer = (String) objs[1];
			} else {
				row = syncWeightsMatrix.getRow(i);
			}

		}
		return buffer;
	}

	private static Object[] performXOR(INDArray in1, INDArray in2) {
		int nColumns = in1.columns();
		StringBuilder resultStringBuilder = new StringBuilder();
		float[] result = new float[nColumns];
		for (int i = 0; i < nColumns; i++) {
			int in1Bit = Integer.valueOf(Math.round(Float.valueOf(in1.getColumn(i).toString())));
			int in2Bit = Integer.valueOf(Math.round(Float.valueOf(in2.getColumn(i).toString())));
			int temp = in1Bit ^ in2Bit;
			result[i] = temp;
			resultStringBuilder.append(temp);
		}
		INDArray resultNDArray = Nd4j.create(result);
		return new Object[] { resultNDArray, resultStringBuilder.toString() };
	}
	
	public static void fitOptimalWeightVector(MultiLayerNetwork model) {
		INDArray weightVector = applyFitnessFunction(model);
		performCrossover(weightVector);
		performMutation(weightVector);
	}
	
	public static INDArray applyFitnessFunction(MultiLayerNetwork model) {
		INDArray weightsMatrix = getWeightMatrix(model);
		int nWeightsMatrixRows = weightsMatrix.rows();
		// INDArray[] wij = new INDArray[nWeightsMatrixRows];
		for (int i = 0; i < nWeightsMatrixRows; i++) {
			INDArray weightRow = weightsMatrix.getRow(i);
			int nColumns = weightRow.columns();
			for (int j = 0; j < nColumns; j++) {
				double initWeight = Double.valueOf(weightRow.getColumn(j).toString());
				int signumWeight = Utils.sigma(initWeight);
				double fitnessWeightValue = (signumWeight * initWeight) + calculateGxFunc(initWeight);
				weightRow.putScalar(i, fitnessWeightValue);
			}

		}
		return weightsMatrix;
	}

	private static void performCrossover(INDArray weightRows) {
		int rows = weightRows.rows();
		for (int i = 0; i < rows; i++) {
			INDArray row1 = weightRows.getRow(i);
			i = i + 1;
			INDArray row2 = weightRows.getRow(i);
			int nColumns = row1.columns();
			for (int j = nColumns / 2; j < nColumns; j++) {
				double w1 = Double.valueOf(row1.getColumn(j).toString());
				double w2 = Double.valueOf(row2.getColumn(j).toString());
				row1.putScalar(j, w2);
				row2.putScalar(j, w1);
			}
		}
	}

	private static void performMutation(INDArray weightRows) {
		int rows = weightRows.rows();
		for (int i = 0; i < rows; i++) {
			INDArray row1 = weightRows.getRow(i);
			int nColumns = row1.columns();
			Random random = new Random(1);
			int index = random.nextInt();
			for (int j = 0; j < nColumns; j++) {
				if (j == index) {
					double w1 = Double.valueOf(row1.getColumn(j).toString());
					w1 = w1 + j;
					row1.putScalar(j, w1);
				}
			}
		}
	}
	
	private static INDArray getWeightMatrix(MultiLayerNetwork model) {
		INDArray weights = model.getLayer(0).getParam(WEIGHT_MATRIX_KEY);
		return weights;
	}
	
	private static double calculateGxFunc(double x) {
		if (x < MIN_WEIGHT_RANGE) {
			return 0;
		} else if (x >= MIN_WEIGHT_RANGE && x < MAX_WEIGHT_RANGE) {
			return x * x;
		} else {
			return 0;
		}
	}

}
