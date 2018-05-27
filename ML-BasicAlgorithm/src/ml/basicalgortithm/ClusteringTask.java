package ml.basicalgortithm;
/**
 * 
 */

/**
 * @XinCheng 2018年5月15日 Administrator
 * 通过分析银行数据集实现基于Weka的聚类模型(期望最大化聚类算法)识别常见客户组，评估其性能
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;

public class ClusteringTask {
	public static void main(String[] args) throws Exception {
		// 加载数据（数据集包含600个实例，每个实例用11个属性描述）
		Instances data = new Instances(new BufferedReader(new FileReader("data/bank-data.arff")));

		// 聚类器的新实例 （EM(Expectation Maximization期望最大化)聚类算法）
		// EM中使用的簇数，可以手动设置，也可通过交叉验证自动设置，可使用肘部法则(elbow-method)确定--该方法会查看特定簇数所解释的偏差百分比
		EM model = new EM();
		// 创建聚类器
		model.buildClusterer(data);
		System.out.println(model);

		// 使用对数似然估量(log likelihood measure)评估聚类算法质量--测量被识别的簇的一致程度
		double logLikelihood = ClusterEvaluation.crossValidateModel(model, data, 10, new Random(1)); // 数据集划分成10个折(folds),针对每个折运行聚类(若聚类算法为相似数据给出高概率，则其在捕获数据结构方面可能做的更好)
		System.out.println(logLikelihood);
	}

}
