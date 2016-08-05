import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.*;
import java.util.*;
import java.net.*;

public class ConnectivityMap {
	public int SAMPLE_COUNT = 1650; 
	public int GENE_COUNT = 11350; 

	private double[] referenceScores;
	
	private double[][] groundTruth;
	private double[][] predictions;
	private double[][] correlation;
	private double[] fracRanks;
	
public double[] Scale(double[] x) {
	double sumX = 0;
	double sumX2 = 0;
	int n = x.length;
	for (int i = 0; i < x.length; i++) {
		sumX += x[i];
		sumX2 += x[i] * x[i];
	}
	double mean = sumX / n;
	double sd = Math.sqrt((n * sumX2 - sumX * sumX) / (n * (n - 1)));
	double[] ret = new double[x.length];
	for (int i = 0; i < x.length; i++) {
		ret[i] = sd == 0 ? 0 : (x[i] - mean) / sd;
	}
	return ret;
}

public double[] Rank(double[] x) {
	double[] ret = new double[x.length];
	double[] d = Arrays.copyOf(x, x.length);
	Arrays.sort(d);
	HashMap<Double, Double> ranks = new HashMap<Double, Double>();
	for (int i = 0; i < x.length; i++) {
		if (ranks.containsKey(d[i])) {
			ranks.put(d[i], ranks.get(d[i]) + 0.5);
		} else {
			ranks.put(d[i], i + 1.0);
		}
	}
	for (int i = 0; i < x.length; i++) {
		ret[i] = ranks.get(x[i]);
	}

	return ret;
}


public double FracRank(double[] x, double y) {
	double val = 0.5;
	for (int i = 0; i < x.length; i++) {
		if (y > x[i]) {
			val += 1.0;
		} else if (y == x[i]) {
			val += 0.5;
		}
	}
	return val / x.length;	
}

public double FracRankColumn(int c) {
	double ret = 0.5;
	double value = correlation[c][c];
	for (int i = 0; i < GENE_COUNT; i++) {
		if (value > correlation[i][c]) {
			ret += 1.0;
		} else if (value == correlation[i][c]) {
			ret += 0.5;
		}
	}
	return ret / GENE_COUNT;	
}

public double SpearmanCorrelation(double[] x, double[] y) {
	int n = x.length;
	if (n == 0) return 0;
	double ret = 0;
	for (int i = 0; i < x.length; i++) {
		ret += x[i] * y[i];
	}
	ret /= (n - 1);
	return ret;
}

	private void loadGroundTruth() {
		for (int i = 0; i < GENE_COUNT; i++) {
			groundTruth[i] = Scale(Rank(groundTruth[i]));
		}
	}
	
	private void calculateCorrelations() {
		for (int i = 0; i < GENE_COUNT; i++) {
			for (int j = 0; j < GENE_COUNT; j++) {
				correlation[i][j] = SpearmanCorrelation(groundTruth[i], predictions[j]);
			}
		}
	}
	
	private void calculateFracRanks() {
		for (int i = 0; i < GENE_COUNT; i++) {
			fracRanks[i] = FracRankColumn(i);
		}
	}
	
	private double[][] loadMatrix(String fileName) throws Exception {
		Path path = Paths.get(fileName);
		Charset charset = Charset.forName("ISO-8859-1");
		List<String> lines = Files.readAllLines(path, charset);
		double[][] ret = new double[lines.size()][];
		for (int i = 0; i < ret.length; i++) {
			String[] s = lines.get(i).split(",");
			ret[i] = new double[s.length];
			for (int j = 0; j < s.length; j++)
			  ret[i][j] = Double.parseDouble(s[j]);
		}
		return ret;
	}
	
	private double[] loadReferenceScores(String filename) throws Exception {
		double[][] raw = loadMatrix(filename);
		double[] ret = new double[raw.length];
		for (int i = 0; i < ret.length; i++) ret[i] = raw[i][0];
		return ret;
	}

	public double score(String predictionFile, String truthFile, String scoreFile) {
		try {
			referenceScores = loadReferenceScores(scoreFile);
			groundTruth = loadMatrix(truthFile);
			predictions = loadMatrix(predictionFile);
			SAMPLE_COUNT = groundTruth[0].length;
			GENE_COUNT = groundTruth.length;
			if (predictions.length != GENE_COUNT) {
				System.out.println("Number of genes in predictions does not match truth.");
				return -1.0;
			}
			correlation = new double[GENE_COUNT][GENE_COUNT];
			fracRanks = new double[GENE_COUNT];

			int geneId = 0;
			while (geneId < predictions.length) {
				if (predictions[geneId].length != SAMPLE_COUNT) {
					System.out.println("Line " + (geneId + 1) + " of the prediction file had " + predictions[geneId].length + " elements, expected " + SAMPLE_COUNT);
					return -1.0;
				}
				predictions[geneId] = Scale(Rank(predictions[geneId]));
				geneId++;
			}

			if (geneId < GENE_COUNT) {
				System.out.println("Prediction file only had " + geneId + " lines, expected " + GENE_COUNT);
				return -1.0;
			}
			
			System.out.println("Prediction file parsed successfully.");
			
			loadGroundTruth();
			calculateCorrelations();
			calculateFracRanks();
			
			double ret = 0;
						
			for (int i = 0; i < GENE_COUNT; i++) {
				double rawScore = (Math.max(correlation[i][i], 0) + fracRanks[i]) * 0.5;

				ret += 1000000.0 * (2.0 - referenceScores[i]) / (2.0 - rawScore);
			}
			
			ret /= GENE_COUNT;
			
			// ret *= (scoreCeiling[testCase] - 1000000) / (scoreCeiling[testCase] - ret);

			if (ret != ret) {
				System.out.println("NaN value detected");
				return -1.0;
			}
						
			return ret;
			
		} catch (Exception e) {
			StringWriter sw = new StringWriter();
			e.printStackTrace(new PrintWriter(sw));
			System.out.println(sw.toString());
			System.out.println("If you see this message, then something went seriously wrong while your solution was tested. Please contact admins.");
			return -1.0;
		}
	}

	public static void main(String[] args) {
		if (args.length < 3) {
			System.out.println("Usage:\n");
			System.out.println("java ConnectivityMap <predictionFile> <truthFile> <scoreFile>");
			return;
		}
		ConnectivityMap instance = new ConnectivityMap();
		double score = instance.score(args[0], args[1], args[2]);
		System.out.println("Final score = " + score);
	}
}
