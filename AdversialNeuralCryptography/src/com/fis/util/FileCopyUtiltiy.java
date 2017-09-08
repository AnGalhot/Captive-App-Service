package com.fis.util;



import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class FileCopyUtiltiy {

	public static void main(String[] args) {
		//load("D:/Backup/TEMP/Keystore/label-generator.txt", "  ");
		generateTrainData();
		generateLabelData();
		System.out.println("Done !!!");
	}
	
	/*public static void generateTrainData() {
		load("D:/Backup/TEMP/Keystore/dummy.txt", "  ","D:/Backup/TEMP/Keystore/ABC.txt");
	}
	
	public static void generateLabelData() {
		load("D:/Backup/TEMP/Keystore/label-generator.txt", "  ","D:/Backup/TEMP/Keystore/GeneratedLabel.txt");
	}*/
	
	public static void generateTrainData() {
		load("D:/Backup/TEMP/Keystore/dummy.txt", "  ","D:/Backup/Innovation In 48 2017/Berlin/Temp/ABC.txt");
	}
	
	public static void generateLabelData() {
		load("D:/Backup/Innovation In 48 2017/Berlin/Temp/alphabet-labels.txt", "  ","D:/Backup/Innovation In 48 2017/Berlin/Temp/GeneratedLabel.txt");
	}

	public static void load(String filePath, String tag,String outPath) {
		double[][] inputs = new double[100][784];
		double[][] labels = new double[100][10];
		File file = new File(filePath);
		DataSetIterator dsItr = null;
		try {

			BufferedReader in = new BufferedReader(new FileReader(file));
			PrintWriter writer = new PrintWriter(new FileWriter(outPath));
			String line1 = null;
			String line2 = null;
			String line = null;
			int count = 0;
			line1 = in.readLine();
			line2 = in.readLine();
			String line3 = in.readLine();
			String line4 = in.readLine();
			String line5 = in.readLine();
			String line6 = in.readLine();
			String line7 = in.readLine();
			String line8 = in.readLine();
			String line9 = in.readLine();
			String line10 = in.readLine();
			String line11 = in.readLine();
			String line12 = in.readLine();
			String line13 = in.readLine();
			String line14 = in.readLine();
			String line15 = in.readLine();
			String line16 = in.readLine();
			String line17 = in.readLine();
			String line18 = in.readLine();
			String line19 = in.readLine();
			String line20 = in.readLine();
			String line21 = in.readLine();
			String line22 = in.readLine();
			String line23 = in.readLine();
			String line24 = in.readLine();
			String line25 = in.readLine();
			String line26 = in.readLine();
			String line27 = in.readLine();
			String line28 = in.readLine();
			String line29 = in.readLine();
			String line30 = in.readLine();
			String line31 = in.readLine();
			String line32 = in.readLine();
			String line33 = in.readLine();
			String line34 = in.readLine();
			String line35 = in.readLine();
			String line36 = in.readLine();
			while (count < 4000) {
				writer.write(line1 + "\n");
				count++;
				writer.write(line2 + "\n");
				count++;
				writer.write(line3 + "\n");
				count++;
				writer.write(line4 + "\n");
				count++;
				writer.write(line5 + "\n");
				count++;
				writer.write(line6 + "\n");
				count++;
				writer.write(line7 + "\n");
				count++;
				writer.write(line8 + "\n");
				count++;
				writer.write(line9 + "\n");
				count++;
				writer.write(line10 + "\n");
				count++;
				writer.write(line11 + "\n");
				count++;
				writer.write(line12 + "\n");
				count++;
				writer.write(line13 + "\n");
				count++;
				writer.write(line14 + "\n");
				count++;
				writer.write(line15 + "\n");
				count++;
				writer.write(line16 + "\n");
				count++;
				writer.write(line17 + "\n");
				count++;
				writer.write(line18 + "\n");
				count++;
				writer.write(line19 + "\n");
				count++;
				writer.write(line20 + "\n");
				count++;
				writer.write(line21 + "\n");
				count++;
				writer.write(line22 + "\n");
				count++;
				writer.write(line23 + "\n");
				count++;
				writer.write(line24 + "\n");
				count++;
				writer.write(line25 + "\n");
				count++;
				writer.write(line26 + "\n");
				count++;
				writer.write(line27 + "\n");
				count++;
				writer.write(line28 + "\n");
				count++;
				writer.write(line29 + "\n");
				count++;
				writer.write(line30 + "\n");
				count++;
				writer.write(line31 + "\n");
				count++;
				writer.write(line32 + "\n");
				count++;
				writer.write(line33 + "\n");
				count++;
				writer.write(line34 + "\n");
				count++;
				writer.write(line35 + "\n");
				count++;
				writer.write(line36 + "\n");
				count++;
							}
			while(count < 18000) {
				writer.write(line1 + "\n");
				count++;
				writer.write(line1 + "\n");
				count++;
				writer.write(line2 + "\n");
				count++;
				writer.write(line2 + "\n");
				count++;
				writer.write(line3 + "\n");
				count++;
				writer.write(line3 + "\n");
				count++;
				writer.write(line4 + "\n");
				count++;
				writer.write(line4 + "\n");
				count++;
				writer.write(line5 + "\n");
				count++;
				writer.write(line5 + "\n");
				count++;
				writer.write(line6 + "\n");
				count++;
				writer.write(line6 + "\n");
				count++;
				writer.write(line7 + "\n");
				count++;
				writer.write(line7 + "\n");
				count++;
				writer.write(line8 + "\n");
				count++;
				writer.write(line8 + "\n");
				count++;
				writer.write(line9 + "\n");
				count++;
				writer.write(line9 + "\n");
				count++;
				writer.write(line10 + "\n");
				count++;
				writer.write(line10 + "\n");
				count++;
				writer.write(line11 + "\n");
				count++;
				writer.write(line11 + "\n");
				count++;
				writer.write(line12 + "\n");
				count++;
				writer.write(line12 + "\n");
				count++;
				writer.write(line13 + "\n");
				count++;
				writer.write(line13 + "\n");
				count++;
				writer.write(line14 + "\n");
				count++;
				writer.write(line14 + "\n");
				count++;
				writer.write(line15 + "\n");
				count++;
				writer.write(line15 + "\n");
				count++;
				writer.write(line16 + "\n");
				count++;
				writer.write(line16 + "\n");
				count++;
				writer.write(line17 + "\n");
				count++;
				writer.write(line17 + "\n");
				count++;
				writer.write(line18 + "\n");
				count++;
				writer.write(line18 + "\n");
				count++;
				writer.write(line19 + "\n");
				count++;
				writer.write(line19 + "\n");
				count++;
				writer.write(line20 + "\n");
				count++;
				writer.write(line20 + "\n");
				count++;
				writer.write(line21 + "\n");
				count++;
				writer.write(line21 + "\n");
				count++;
				writer.write(line22 + "\n");
				count++;
				writer.write(line22 + "\n");
				count++;
				writer.write(line23 + "\n");
				count++;
				writer.write(line23 + "\n");
				count++;
				writer.write(line24 + "\n");
				count++;
				writer.write(line24 + "\n");
				count++;
				writer.write(line25 + "\n");
				count++;
				writer.write(line25 + "\n");
				count++;
				writer.write(line26 + "\n");
				count++;
				writer.write(line26 + "\n");
				count++;
				writer.write(line27 + "\n");
				count++;
				writer.write(line27 + "\n");
				count++;
				writer.write(line28 + "\n");
				count++;
				writer.write(line28 + "\n");
				count++;
				writer.write(line29 + "\n");
				count++;
				writer.write(line29 + "\n");
				count++;
				writer.write(line30 + "\n");
				count++;
				writer.write(line30 + "\n");
				count++;
				writer.write(line31 + "\n");
				count++;
				writer.write(line31 + "\n");
				count++;
				writer.write(line32 + "\n");
				count++;
				writer.write(line32 + "\n");
				count++;
				writer.write(line33 + "\n");
				count++;
				writer.write(line33 + "\n");
				count++;
				writer.write(line34 + "\n");
				count++;
				writer.write(line34 + "\n");
				count++;
				writer.write(line35 + "\n");
				count++;
				writer.write(line35 + "\n");
				count++;
				writer.write(line36 + "\n");
				count++;
				writer.write(line36 + "\n");
				count++;
				
			}
			while(count < 24000) {
				writer.write(line1 + "\n");
				count++;
				writer.write(line2 + "\n");
				count++;
				writer.write(line3 + "\n");
				count++;
				writer.write(line4 + "\n");
				count++;
				writer.write(line5 + "\n");
				count++;
				writer.write(line6 + "\n");
				count++;
				writer.write(line7 + "\n");
				count++;
				writer.write(line8 + "\n");
				count++;
				writer.write(line9 + "\n");
				count++;
				writer.write(line10 + "\n");
				count++;
				writer.write(line11 + "\n");
				count++;
				writer.write(line12 + "\n");
				count++;
				writer.write(line13 + "\n");
				count++;
				writer.write(line14 + "\n");
				count++;
				writer.write(line15 + "\n");
				count++;
				writer.write(line16 + "\n");
				count++;
				writer.write(line17 + "\n");
				count++;
				writer.write(line18 + "\n");
				count++;
				writer.write(line19 + "\n");
				count++;
				writer.write(line20 + "\n");
				count++;
				writer.write(line21 + "\n");
				count++;
				writer.write(line22 + "\n");
				count++;
				writer.write(line23 + "\n");
				count++;
				writer.write(line24 + "\n");
				count++;
				writer.write(line25 + "\n");
				count++;
				writer.write(line26 + "\n");
				count++;
				writer.write(line27 + "\n");
				count++;
				writer.write(line28 + "\n");
				count++;
				writer.write(line29 + "\n");
				count++;
				writer.write(line30 + "\n");
				count++;
				writer.write(line31 + "\n");
				count++;
				writer.write(line32 + "\n");
				count++;
				writer.write(line33 + "\n");
				count++;
				writer.write(line34 + "\n");
				count++;
				writer.write(line35 + "\n");
				count++;
				writer.write(line36 + "\n");
				count++;
				
			}
			in.close();
			writer.flush();
			writer.close();

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
