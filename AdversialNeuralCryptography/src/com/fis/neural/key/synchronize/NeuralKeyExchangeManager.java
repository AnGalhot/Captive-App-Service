package com.fis.neural.key.synchronize;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * 
 * This class represents the mechanism to generate neural key exchange using
 * weight synchronization based on Diffe-Hellman protocol Diffie–Hellman Key
 * Exchange establishes a shared secret between two parties that can be used for
 * secret communication for exchanging data over a public network.
 *
 */
public class NeuralKeyExchangeManager {
	private static final int seed = 12345;
	private static final Random random = new Random(seed);

	private enum EndPointType {
		Transmitter("Transmitter"), Receiver("Receiver");

		private String name;

		EndPointType(String name) {
			this.name = name;
		}

		public String toString() {
			return this.name;
		}
	}

	public static void main(String arr[]) {

		NeuralKeyExchangeManager m = new NeuralKeyExchangeManager();
		m.generateKeyByWeightSnchronization();
	}

	/**
	 * This method is used to generate private key using weight synchronization
	 * of neural networks between two respective parties.
	 * 
	 * @return key
	 */
	public Object generateKeyByWeightSnchronization() {

		// build transmitter neural net0
		ConveyorKeySynchronizeNeuralNet transmitterNN = (ConveyorKeySynchronizeNeuralNet) constructNeuralNet(
				EndPointType.Transmitter);
		// build receiver neural net
		ReceiverKeySynchronizeNeuralNet receiverNN = (ReceiverKeySynchronizeNeuralNet) constructNeuralNet(
				EndPointType.Receiver);

		//String key = generateKey(transmitterNN, receiverNN);
		String key = generateKeyUsingGenericAlgo(transmitterNN, receiverNN);
		System.out.println(key);
		return key;
	}

	private String generateKey(ConveyorKeySynchronizeNeuralNet transmitterNeuralNet,
			ReceiverKeySynchronizeNeuralNet receiverNeuralNet) {
		boolean isSynch = false;
		double transmittalNNScore = 0;
		double receiverNNScore = 0;
		while (!isSynch) {
			// generate random input vector
			Object[] objects = com.fis.neural.key.synchronize.Utils.generateRandomInputVector(random);
			INDArray input = (INDArray) objects[0];
			INDArray label = (INDArray) objects[1];
			transmittalNNScore = transmitterNeuralNet.computeOutput(input, label);
			receiverNNScore = receiverNeuralNet.computeOutput(input, label);
			isSynch = Utils.areSameOutput(transmittalNNScore, receiverNNScore);
		}
		Utils.updateWeightsUsingHebbianLearningRule(transmitterNeuralNet.getModel(), transmittalNNScore,
				receiverNNScore);
		Utils.updateWeightsUsingHebbianLearningRule(receiverNeuralNet.getModel(), transmittalNNScore,
				receiverNNScore);
		String generatedKey = Utils.generateKeyFromWeights(transmitterNeuralNet.getModel());
		return generatedKey;
	}
	
	private String generateKeyUsingGenericAlgo(ConveyorKeySynchronizeNeuralNet transmitterNeuralNet,
			ReceiverKeySynchronizeNeuralNet receiverNeuralNet) {
		boolean isSynch = false;
		double transmittalNNScore = 0;
		double receiverNNScore = 0;
		while (!isSynch) {
			// generate random input vector
			Object[] objects = com.fis.neural.key.synchronize.Utils.generateRandomInputVector(random);
			INDArray input = (INDArray) objects[0];
			INDArray label = (INDArray) objects[1];
			Utils.fitOptimalWeightVector(transmitterNeuralNet.getModel());
			transmittalNNScore = transmitterNeuralNet.computeOutput(input, label);
			Utils.fitOptimalWeightVector(receiverNeuralNet.getModel());
			receiverNNScore = receiverNeuralNet.computeOutput(input, label);
			isSynch = Utils.areSameOutput(transmittalNNScore, receiverNNScore);
		}
		Utils.updateWeightsUsingHebbianLearningRule(transmitterNeuralNet.getModel(), transmittalNNScore,
				receiverNNScore);
		Utils.updateWeightsUsingHebbianLearningRule(receiverNeuralNet.getModel(), transmittalNNScore,
				receiverNNScore);
		String generatedKey = Utils.generateKeyFromWeights(transmitterNeuralNet.getModel());
		return generatedKey;
	}

	/*
	 * private INDArray generateKey(ConveyorKeySynchronizeNeuralNet
	 * transmitterNeuralNet, ReceiverKeySynchronizeNeuralNet receiverNeuralNet)
	 * { boolean isSynch = false; DataSetIterator dsItr = null; double
	 * transmittalNNScore = 0; double receiverNNScore = 0; INDArray receiverRes
	 * = null; INDArray conveyorRes = null; while (!isSynch) { // generate
	 * random input vector Object[] objects =
	 * com.fis.neural.key.synchronize.Utils.generateRandomInputVector(random);
	 * INDArray input = (INDArray) objects[0]; INDArray label = (INDArray)
	 * objects[1]; dsItr = (DataSetIterator) objects[2]; //transmittalNNScore =
	 * transmitterNeuralNet.computeOutput(input, label); conveyorRes =
	 * transmitterNeuralNet.computeWeight(input, label); //receiverNNScore =
	 * receiverNeuralNet.computeOutput(input, label); receiverRes =
	 * receiverNeuralNet.computeWeight(input, label); //isSynch =
	 * Utils.areSameScore(transmittalNNScore, receiverNNScore); isSynch =
	 * Utils.areSameScore(conveyorRes, receiverRes); } //
	 * transmitterNeuralNet.updateWeightsUsingHebbianLearningRule(
	 * transmittalNNScore); return null; }
	 */

	private Object constructNeuralNet(EndPointType type) {
		switch (type) {
		case Transmitter: {
			ConveyorKeySynchronizeNeuralNet transmitterNeuralNet = createTransmitterNNInstance();
			return transmitterNeuralNet;
		}
		case Receiver: {
			ReceiverKeySynchronizeNeuralNet receiverNeuralNet = createReceiverNNInstance();
			return receiverNeuralNet;
		}
		default: {
			throw new IllegalArgumentException("Please specify valid value of EndPoint type... ");
		}
		}
	}

	private ConveyorKeySynchronizeNeuralNet createTransmitterNNInstance() {
		return ConveyorKeySynchronizeNeuralNet.getInstance();
	}

	private ReceiverKeySynchronizeNeuralNet createReceiverNNInstance() {
		return ReceiverKeySynchronizeNeuralNet.getInstance();
	}

}
