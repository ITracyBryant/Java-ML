/**
 * 
 */
package anomaly.fraud;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.LOF;

/**
 * @XinCheng 2018��5��22�� Administrator
 *
 */
public class Anomaly {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// ��������
		String filePath = "data/ydata/Benchmark/real_"; // �����ļ�����ͬǰ׺�ļ���

		// �ļ���ÿһ�о���һ������ֵ��ʱ������
		List<List<Double>> rawData = new ArrayList<List<Double>>();
		// ��ֱ��ͼ����̬������ʱ��Ҫ�õ�min��maxֵ
		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;
		for (int i = 1; i <= 67; i++) {
			List<Double> sample = new ArrayList<Double>();
			BufferedReader reader = new BufferedReader(new FileReader(filePath + i + ".csv"));

			boolean isAnomaly = false; // �ж��쳣ֵ
			reader.readLine();
			while (reader.ready()) {
				String line[] = reader.readLine().split(",");
				double value = Double.parseDouble(line[1]);
				sample.add(value);
				max = Math.max(max, value);
				min = Double.min(min, value);
				if (line[2] == "1")
					isAnomaly = true;
			}
			System.out.println(isAnomaly);
			reader.close();
			rawData.add(sample);
		}
		System.out.println(rawData.size() + "\nmax: " + max + "\nmin: " + min);

		// ����ֱ��ͼ
		int WIN_SIZE = 500;
		int HIST_BINS = 20;
		int current = 0;

		List<double[]> dataHist = new ArrayList<double[]>();
		for (List<Double> sample : rawData) {
			double[] histogram = new double[HIST_BINS];
			for (double value : sample) {
				int bin = toBin(normalize(value, min, max), HIST_BINS);
				histogram[bin]++;
				current++;
				if (current == WIN_SIZE) {
					current = 0;
					dataHist.add(histogram);
					histogram = new double[HIST_BINS];
				}
			}
			dataHist.add(histogram);
		}

		// ��̬��ֱ��ͼ
		for (double[] histogram : dataHist) {
			int sum = 0;
			for (double v : histogram) {
				sum += v;
			}
			for (int i = 0; i < histogram.length; i++) {
				histogram[i] /= 1.0 * sum;
			}
		}
		System.out.println("Total histograms: " + dataHist.size());

		// ��ֱ��ͼת��ΪWeka�е�Instances����ÿ��ֱ��ͼ��Ӧ��һ��Weka���� Create DB on-the-fly
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < HIST_BINS; i++) {
			attributes.add(new Attribute("Hist-" + i));
		}
		Instances dataset = new Instances("My dataset", attributes, dataHist.size());
		for (double[] histogram : dataHist) {
			dataset.add(new DenseInstance(1.0, histogram));
		}
		System.out.println("Dataset created: " + dataset.size());

		// ����ģ��
		// �����ݼ��ָ��ѵ�����Ͳ��Լ�
		Instances trainData = dataset.testCV(2, 0); // ��һ������ָ���������ڶ�������ָ�������ĸ���
		Instances testData = dataset.testCV(2, 1);
		System.out.println("Train: " + trainData.size() + "\nTest: " + testData.size());

		// ʹ��k-nn�㷨����ѵ�����ݣ�ʹ�üල�����ܶȵ�K������㷨���з��ࣻ��Ϊ�޼ල���������ֲ��쳣�����㷨
		LOF lof = new LOF();
		lof.setInputFormat(trainData);
		lof.setOptions(new String[] { "-min", "3", "-max", "3" }); // LOF����ָ��������ͬ��k������һ�����Ͻ磬һ�����½�
		for (Instance inst : trainData) {
			lof.input(inst); // ������
		}
		lof.batchFinished(); // �ڲ������ʼ��
		System.out.println("LOF loaded");

		Instances testDataLofScore = Filter.useFilter(testData, lof);

		for (Instance inst : testDataLofScore) {
			System.out.println(inst.value(inst.numAttributes() - 1));
		}
	}

	/**
	 * ��̬��ֵ��[0, 1]����
	 * 
	 * @param value
	 * @param min
	 * @param max
	 * @return
	 */
	static double normalize(double value, double min, double max) {
		return value / (max - min);
	}

	/**
	 * returns a bin in range [0, bins). Assumes value is normalized to interval
	 * [0, 1]
	 * 
	 * @param normalizedValue
	 * @param bins
	 * @return
	 */
	static int toBin(double normalizedValue, int bins) {
		if (normalizedValue == 1.0) {
			return bins - 1;
		}
		return (int) (normalizedValue * bins);
	}

}
