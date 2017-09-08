package com.fis.adversarial.cryptography;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class PostEvaluationResultTransformer {

	private PostEvaluationResultTransformer() {

	}

	public static PostEvaluationResultTransformer getInstance() {
		return new PostEvaluationResultTransformer();
	}

	private static final char[] alphanumericChars = new char[] { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
			'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
			'W', 'X', 'Y', 'Z' };

	private enum Rules {
		A("A", "0100000000"), B("B", "0000000001"), C("C", "0001000000"), D("D", "1000000000"), E("E", "0000010000"), F(
				"F", "0000000010"), G("G", "0010000000"), H("H", "0000001000"), I("I", "0000100000"), J("J",
						"0000000100"), K("K", "0110000000"), L("L", "1000000001"), M("M", "0001100000"), N("N",
								"1100000000"), O("O", "0000011000"), P("P", "0000000011"), Q("Q", "0011000000"), R("R",
										"0000001100"), S("S", "0000110000"), T("T", "0000000110"), U("U",
												"0111000000"), V("V", "1100000001"), W("W", "0001110000"), X("X",
														"1110000000"), Y("Y", "0000011100"), Z("Z", "1000000011");
		private String ruleType;
		private int[] mask;

		private Rules(String name, String mask) {
			this.ruleType = name;
			int[] temp = new int[10];
			for (int i = 0; i < 10; i++) {
				temp[i] = Integer.valueOf(mask.substring(i, i + 1));
			}
			this.mask = temp;
		}
	}

	public int[] applyRulesOnOutcome(INDArray input, char ch) {
		INDArray resultArray = input.getRow(0);
		int[] resultBits = convertToIntArray(resultArray);
		Rules charRule = getCharacterRule(ch);
		int[] transformedOutcome = transformOutcomeByRule(charRule, resultBits);
		// INDArray transformedOutcome = Nd4j.create(tempOutcome);
		return transformedOutcome;

	}

	public char applyRulesOnOutcome(INDArray input) {
		INDArray resultArray = input.getRow(0);
		int[] resultbits = convertToIntArray(resultArray);
        return transformOutcomeToCharacter(resultbits);
	}

	private char transformOutcomeToCharacter(int[] inputBits) {
		int index = 0;
		char ch = ' ';
		for (int i : inputBits) {
			if (i == 1) {
				ch = alphanumericChars[index];
				break;
			}
			index++;
		}
		return ch;
	}

	private int[] transformOutcomeByRule(Rules rule, int[] inputBits) {
		int[] result = null;
		switch (rule) {
		case A: {
			result = doTransform(inputBits, Rules.A.mask);
			break;
		}
		case B: {
			result = doTransform(inputBits, Rules.B.mask);
			break;
		}
		case C: {
			result = doTransform(inputBits, Rules.C.mask);
			break;
		}
		case D: {
			result = doTransform(inputBits, Rules.D.mask);
			break;
		}
		case E: {
			result = doTransform(inputBits, Rules.E.mask);
			break;
		}
		case F: {
			result = doTransform(inputBits, Rules.F.mask);
			break;
		}
		case G: {
			result = doTransform(inputBits, Rules.G.mask);
			break;
		}
		case H: {
			result = doTransform(inputBits, Rules.H.mask);
			break;
		}
		case I: {
			result = doTransform(inputBits, Rules.I.mask);
			break;
		}
		case J: {
			result = doTransform(inputBits, Rules.J.mask);
			break;
		}
		case K: {
			result = doTransform(inputBits, Rules.K.mask);
			break;
		}
		case L: {
			result = doTransform(inputBits, Rules.L.mask);
			break;
		}
		case M: {
			result = doTransform(inputBits, Rules.M.mask);
			break;
		}
		case N: {
			result = doTransform(inputBits, Rules.N.mask);
			break;
		}
		case O: {
			result = doTransform(inputBits, Rules.O.mask);
			break;
		}
		case P: {
			result = doTransform(inputBits, Rules.P.mask);
			break;
		}
		case Q: {
			result = doTransform(inputBits, Rules.Q.mask);
			break;
		}
		case R: {
			result = doTransform(inputBits, Rules.R.mask);
			break;
		}
		case S: {
			result = doTransform(inputBits, Rules.S.mask);
			break;
		}
		case T: {
			result = doTransform(inputBits, Rules.T.mask);
			break;
		}
		case U: {
			result = doTransform(inputBits, Rules.U.mask);
			break;
		}
		case V: {
			result = doTransform(inputBits, Rules.V.mask);
			break;
		}
		case W: {
			result = doTransform(inputBits, Rules.W.mask);
			break;
		}
		case X: {
			result = doTransform(inputBits, Rules.X.mask);
			break;
		}
		case Y: {
			result = doTransform(inputBits, Rules.Y.mask);
			break;
		}
		case Z: {
			result = doTransform(inputBits, Rules.Z.mask);
			break;
		}
		default: {
			result = new int[inputBits.length];
		}
		}
		return result;
	}

	private int[] doTransform(int[] inputBits, int[] mask) {
		// float[] result = new float[inputBits.length];
		int[] result = new int[inputBits.length];
		for (int i = 0, length = inputBits.length; i < length; i++) {
			int inputbit = inputBits[i];
			int maskbit = mask[i];
			int temp = inputbit | maskbit;
			result[i] = temp;
		}
		return result;
	}

	private Rules getCharacterRule(char ch) {
		return Rules.valueOf(String.valueOf(ch));
	}

	private int[] convertToIntArray(INDArray input) {
		int nColumns = input.columns();
		int[] intArray = new int[nColumns];
		for (int i = 0; i < nColumns; i++) {
			String tempValue = input.getColumn(i).toString();
			int bitvalue = Math.round(Float.valueOf(tempValue));
			intArray[i] = bitvalue;
		}
		return intArray;
	}

}
