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
 * @XinCheng 2018年5月22日 Administrator
 *
 */
public class Anomaly {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// 加载数据
		String filePath = "data/ydata/Benchmark/real_"; // 加载文件的相同前缀文件名

		// 文件中每一行就是一个带有值的时间序列
		List<List<Double>> rawData = new ArrayList<List<Double>>();
		// 对直方图做正态化处理时，要用到min和max值
		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;
		for (int i = 1; i <= 67; i++) {
			List<Double> sample = new ArrayList<Double>();
			BufferedReader reader = new BufferedReader(new FileReader(filePath + i + ".csv"));

			boolean isAnomaly = false; // 判断异常值
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

		// 创建直方图
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

		// 正态化直方图
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

		// 将直方图转换为Weka中的Instances对象，每个直方图对应于一个Weka属性 Create DB on-the-fly
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < HIST_BINS; i++) {
			attributes.add(new Attribute("Hist-" + i));
		}
		Instances dataset = new Instances("My dataset", attributes, dataHist.size());
		for (double[] histogram : dataHist) {
			dataset.add(new DenseInstance(1.0, histogram));
		}
		System.out.println("Dataset created: " + dataset.size());

		// 构建模型
		// 将数据集分割成训练集和测试集
		Instances trainData = dataset.testCV(2, 0); // 第一个参数指定折数，第二个参数指定返回哪个折
		Instances testData = dataset.testCV(2, 1);
		System.out.println("Train: " + trainData.size() + "\nTest: " + testData.size());

		// 使用k-nn算法加载训练数据，使用监督基于密度的K最近邻算法进行分类；作为无监督过滤器，局部异常因子算法
		LOF lof = new LOF();
		lof.setInputFormat(trainData);
		lof.setOptions(new String[] { "-min", "3", "-max", "3" }); // LOF允许指定两个不同的k参数，一个作上界，一个作下界
		for (Instance inst : trainData) {
			lof.input(inst); // 正例库
		}
		lof.batchFinished(); // 内部计算初始化
		System.out.println("LOF loaded");

		Instances testDataLofScore = Filter.useFilter(testData, lof);

		for (Instance inst : testDataLofScore) {
			System.out.println(inst.value(inst.numAttributes() - 1));
		}
	}

	/**
	 * 正态化值在[0, 1]区间
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
