package com.fis.adversarial.cryptography;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fis.adversarial.receiver.cryptography.ReceiverConvolutionNN;

public class CryptologyFacade {

	private static final Logger log = LoggerFactory.getLogger(CryptologyFacade.class);

	private static final String key = "1001001001";

	public String getEncryptedOutcome(String message) throws Exception {
		String buffer = message;
		if (buffer == null) {
			return null;
		}
		buffer = buffer.replaceAll("[^a-zA-Z0-9]", "");
		if (buffer == null || buffer.isEmpty() || buffer.length() == 0) {
			return null;
		}
		AdversarialNeuralHelper helper = AdversarialNeuralHelper.getInstance();
		ConveyorConvolutionNN convolutionNN = ConveyorConvolutionNN.getInstance();
		StringBuilder encryptedString = new StringBuilder();
		PostEvaluationResultTransformer transformer = PostEvaluationResultTransformer.getInstance();
		// char[] characters = new char[buffer.length()];
		for (int i = 0; i < buffer.length(); i++) {
			char ch = Character.toUpperCase(buffer.charAt(i));
			INDArray normalizedCharValue = helper.doCharacterNormalization(ch);
			INDArray result = convolutionNN.evaluateResult(normalizedCharValue);
			if (Character.isLetter(ch)) {
				int[] transformedOutcome = transformer.applyRulesOnOutcome(result, ch);
				encryptedString.append(performXOR(transformedOutcome, key));
			} else {
				encryptedString.append(performXOR(result, key));
			}
		}
		return encryptedString.toString();
	}

	public String getDecryptedOutcome(String encryptedMessage) throws Exception {
		String[] encryptedInputTokens = tokenizeInput(encryptedMessage, key.length());
		encryptedInputTokens = performXOR(encryptedInputTokens, key);
		ReceiverConvolutionNN convolutionNN = ReceiverConvolutionNN.getInstance();
		StringBuilder decryptedString = new StringBuilder();
		AdversarialNeuralHelper helper = AdversarialNeuralHelper.getInstance();
		PostEvaluationResultTransformer transformer = PostEvaluationResultTransformer.getInstance();
		for (String token : encryptedInputTokens) {
			INDArray normalizedToken = helper.doEncryptedTokenNormalization(token);
			try {
				INDArray result = convolutionNN.evaluateResult(normalizedToken);
				char outcomeChar = transformer.applyRulesOnOutcome(result);
				decryptedString.append(outcomeChar);
			} catch (Exception e) {
				log.error("Exception occured while evaluating the outcome for token " + token, e);
				throw e;
			}
		}
		return decryptedString.toString();
	}

	public String decryptOutcomeByEavesdropping(String encryptedMessage) throws Exception {
		String[] encryptedInputTokens = tokenizeInputRandomly(encryptedMessage);		
		ReceiverConvolutionNN convolutionNN = ReceiverConvolutionNN.getInstance();
		StringBuilder decryptedString = new StringBuilder();
		AdversarialNeuralHelper helper = AdversarialNeuralHelper.getInstance();
		PostEvaluationResultTransformer transformer = PostEvaluationResultTransformer.getInstance();
		for (String token : encryptedInputTokens) {
			INDArray normalizedToken = helper.doEncryptedTokenNormalization(token);
			try {
				INDArray result = convolutionNN.evaluateResult(normalizedToken);
				char outcomeChar = transformer.applyRulesOnOutcome(result);
				decryptedString.append(outcomeChar);
			} catch (Exception e) {
				log.error("Exception occured while evaluating the outcome for token " + token, e);
				throw e;
			}
		}
		return decryptedString.toString();
	}

	private String[] tokenizeInput(String encryptedMessage, int keyLength) {
		int nChars = (encryptedMessage.length() / keyLength);
		String token = null;
		String[] tokens = new String[nChars];
		int beginIndex = 0;
		for (int i = 0; i < nChars; i++) {
			token = encryptedMessage.substring(beginIndex, beginIndex + keyLength);
			beginIndex = beginIndex + keyLength;
			tokens[i] = token;
		}
		return tokens;
	}

	private String[] tokenizeInputRandomly(String encryptedMessage) {
		int randomNum = (int)(Math.random() * (16 - 10)) + 10;
		int nChars = (encryptedMessage.length() / randomNum);
		String token = null;
		String[] tokens = new String[nChars];
		int beginIndex = 0;
		for (int i = 0; i < nChars; i++) {
			token = encryptedMessage.substring(beginIndex, beginIndex + randomNum);
			beginIndex = beginIndex + randomNum;
			tokens[i] = token;
		}
		return tokens;
	}

	private StringBuilder performXOR(INDArray firstOperand, String key) {
		INDArray resultArray = firstOperand.getRow(0);
		StringBuilder builder = new StringBuilder();
		int[] resultBits = convertToIntArray(resultArray);
		for (int i = 0; i < 10; i++) {
			int resultBit = resultBits[i];
			int keyBit = Integer.valueOf(key.substring(i, i + 1));
			int temp = resultBit ^ keyBit;
			builder.append(String.valueOf(temp));
		}
		return builder;

	}

	private String[] performXOR(String[] encryptedInput, String key) {
		String[] input = new String[encryptedInput.length];
		int index = 0;
		for (String token : encryptedInput) {
			StringBuilder builder = new StringBuilder();
			for (int i = 0, length = token.length(); i < length; i++) {
				int tokenbit = Integer.valueOf(token.substring(i, i + 1));
				int keybit = Integer.valueOf(key.substring(i, i + 1));
				int temp = tokenbit ^ keybit;
				builder.append(String.valueOf(temp));
			}
			input[index++] = builder.toString();
		}
		return input;
	}

	private StringBuilder performXOR(int[] outcomeBits, String key) {
		// INDArray resultArray = firstOperand.getRow(0);
		StringBuilder builder = new StringBuilder();
		int[] resultBits = outcomeBits;
		for (int i = 0; i < 10; i++) {
			int resultBit = resultBits[i];
			int keyBit = Integer.valueOf(key.substring(i, i + 1));
			int temp = resultBit ^ keyBit;
			builder.append(String.valueOf(temp));
		}
		return builder;

	}

	private int[] convertToIntArray(INDArray input) {
		int[] intArray = new int[10];
		for (int i = 0; i < 10; i++) {
			intArray[i] = input.getInt(i);
		}
		return intArray;
	}

	public static void main(String arr[]) throws Exception {
		CryptologyFacade facade = new CryptologyFacade();
		// String encryptedString = facade.getEncryptedOutcome("ANK");
		// System.out.println(encryptedString);
		String decryptedString = facade.getDecryptedOutcome("010100100101010010000111001001");
		System.out.println(decryptedString);
	}

}
