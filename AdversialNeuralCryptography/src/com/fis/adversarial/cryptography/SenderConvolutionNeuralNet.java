package com.fis.adversarial.cryptography;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

public class SenderConvolutionNeuralNet {

	private DataSet loadTrainData(String path, String delimiter) throws IOException, InterruptedException {
		/*Schema inpuitDataSchema = new Schema.Builder().addColumnsString("Char").addColumnsDouble("ASCII_CODE","PLAIN_TXT_BITS","CIPHER_TXT_BITS").build();
		TransformProcess tp = new TransformProcess.Builder(inpuitDataSchema).removeColumns("Char","ASCII_CODE").build();
		FileRecordReader r = new FileRecordReader().nextRecord().getMetaData().*/
		RecordReader recordReader = new CSVRecordReader(1, " ");
		recordReader.initialize(new FileSplit(new File(path)));
		RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 40);
		DataSet dataSet = iterator.next();
		
		DataNormalization normalizer = new NormalizerMinMaxScaler(-1, 1);
		normalizer.fit(dataSet);
		normalizer.transform(dataSet);
		INDArray inputs = dataSet.getFeatureMatrix();
		INDArray labels = dataSet.getLabels();
		return dataSet;
	}
	
	public static void main(String[] args) throws IOException, InterruptedException {
		SenderConvolutionNeuralNet n = new SenderConvolutionNeuralNet();
		n.loadTrainData("D:\\Hacathon-2017\\Berlin\\ANCDataSet.csv", " ");
	}

}
