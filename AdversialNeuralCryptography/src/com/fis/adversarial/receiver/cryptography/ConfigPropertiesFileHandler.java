package com.fis.adversarial.receiver.cryptography;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;
import java.util.Set;

public class ConfigPropertiesFileHandler {

	private static Map<String, String> cacheConfigurations = new HashMap<String, String>(36);
	private static final Logger log = LoggerFactory.getLogger(ConfigPropertiesFileHandler.class);
	private static final String propertiesFilePath = "config/receiver-config.properties";
	private static ConfigPropertiesFileHandler INSTANCE = null;

	private ConfigPropertiesFileHandler() {
		loadProperties();
	}

	private void loadProperties() {
		Properties prop = new Properties();
		InputStream input = null;
		try {
			input = new FileInputStream(propertiesFilePath);
			// load a properties file
			prop.load(input);
			Set<Entry<Object, Object>> entries = prop.entrySet();
			for (Entry<Object, Object> entry : entries) {
				cacheConfigurations.put(String.valueOf(entry.getKey()), String.valueOf(entry.getValue()));
			}

		} catch (IOException io) {
			log.error("IOException occurred while loading properties file", io);
		} catch (Exception e) {
			log.error("IOException occurred while loading properties file", e);
		} finally {
			if (input != null) {
				try {
					input.close();
				} catch (IOException e) {
					log.error("Fatal error occured while closing the proeprties file input stream...");
				}
			}
		}
	}

	public static ConfigPropertiesFileHandler getInstance() {
		if (INSTANCE == null) {
			INSTANCE = new ConfigPropertiesFileHandler();
		}
		return INSTANCE;
	}

	public String getPropertyValue(String propertyName) {
		String value = cacheConfigurations.get(propertyName);
		return value;
	}
	
	public INDArray getNormalizedValue(String propertyName) {
		String value = getPropertyValue(propertyName);
		String[] datas = value.split("  ");
		double[] data = new double[datas.length];
		for (int i = 0; i < datas.length; i++)
			data[i] = Double.parseDouble(datas[i]);
		double[][] normalizedValue = new double[1][0];
		normalizedValue[0] = data;
		INDArray normalizedValueArray = Nd4j.create(normalizedValue);
		return normalizedValueArray;
	}

}
