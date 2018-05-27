/**
 * 
 */
package anomaly.fraud;

import java.io.File;
import java.util.ArrayList;
import java.util.ListIterator;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

/**
 * @XinCheng 2018年5月22日 Administrator 为可疑模式建模，保险理赔欺诈检测
 */
public class Fraud {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		String filePath = "data/claims.csv";

		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File(filePath));
		Instances data = loader.getDataSet();

		// 配置数据集
		// 属性索引
		int CLASS_INDEX = 15;
		int POLICY_INDEX = 17;
		@SuppressWarnings("unused")
		int NO_FRAUD = 0, FRAUD = 1;
		int FOLDS = 3;

		// 确保所有属性类型都是名义型
		NumericToNominal toNominal = new NumericToNominal();
		toNominal.setInputFormat(data);
		data = Filter.useFilter(data, toNominal);

		// 指定待预测的属性，设置对应的类索引
		data.setClassIndex(CLASS_INDEX);

		// 移除一个描述保单编号的属性(policy attribute)
		Remove remove = new Remove();
		remove.setInputFormat(data);
		remove.setOptions(new String[] { "-R", "" + POLICY_INDEX });
		data = Filter.useFilter(data, remove);

		System.out.println(data.toSummaryString());
		System.out.println("分类的类别属性：\n" + data.attributeStats(data.classIndex()));

		// Vanilla approach 纯方法--既不需要做数据预处理，也不需要考虑数据集具体细节，用k折交叉验证
		// 定义分类器
		ArrayList<Classifier> models = new ArrayList<Classifier>();
		models.add(new J48());
		models.add(new RandomForest());
		models.add(new NaiveBayes());
		models.add(new AdaBoostM1());
		models.add(new Logistic());

		Evaluation eval = new Evaluation(data);
		System.out.println("Vanilla approach\n-------------------");
		for (Classifier model : models) {
			eval.crossValidateModel(model, data, FOLDS, new Random(1), new Object[] {});
			System.out.println(model.getClass().getName() + "\n" + "\tRecall:    " + eval.recall(FRAUD) + "\n"
					+ "\tPrecision: " + eval.precision(FRAUD) + "\n" + "\tF-measure: " + eval.fMeasure(FRAUD));
		}

		// 数据集重整(Dataset
		// rebalancing),手工实现k折交叉验证。Weka内置过滤器Resample用于从一个数据集中随机抽取子样本，使用重置抽样或不重置抽样，可以将一个分布调整为类均匀分布
		// 使用StratifiedRemoveFolds过滤器划分数据集，划分后的各折中依然保持相同的类分布
		StratifiedRemoveFolds kFold = new StratifiedRemoveFolds();
		kFold.setInputFormat(data);
		double measures[][] = new double[models.size()][FOLDS];
		for (int k = 1; k <= FOLDS; k++) {
			// 把数据划分成测试折和训练折
			kFold.setOptions(new String[] { "-N", "" + FOLDS, "-F", "" + k, "-S", "1" });
			Instances test = Filter.useFilter(data, kFold);
			kFold.setOptions(new String[] { "-N", "" + FOLDS, "-F", "" + k, "-S", "1", "-V" });// 反选
																								// 参数"-V"
			Instances train = Filter.useFilter(data, kFold);
			System.out.println("Fold" + k + "\n\ttrain: " + train.size() + "\n\ttest: " + test.size());

			// 重整训练集，-Z参数指定要重抽样数据集的比例，-B参数把类分布调整为均匀分布
			Resample resample = new Resample();
			resample.setInputFormat(data);
			resample.setOptions(new String[] { "-Z", "100", "-B", "1" }); // 重整替换
			Instances balancedTrain = Filter.useFilter(train, resample);

			// 创建分类器，做评估
			for (ListIterator<Classifier> it = models.listIterator(); it.hasNext();) {
				Classifier model = it.next();
				model.buildClassifier(balancedTrain);
				eval = new Evaluation(balancedTrain);
				eval.evaluateModel(model, test);
				// System.out.println(
				// "\n\t"+model.getClass().getName() + "\n"+
				// "\tRecall: "+eval.recall(FRAUD) + "\n"+
				// "\tPrecision: "+eval.precision(FRAUD) + "\n"+
				// "\tF-measure: "+eval.fMeasure(FRAUD));

				// 为计算平均数保存结果
				measures[it.previousIndex()][0] += eval.recall(FRAUD);
				measures[it.previousIndex()][1] += eval.precision(FRAUD);
				measures[it.previousIndex()][2] += eval.fMeasure(FRAUD);
			}
		}

		// 计算平均数
		for (int i = 0; i < models.size(); i++) {
			measures[i][0] /= 1.0 * FOLDS;
			measures[i][1] /= 1.0 * FOLDS;
			measures[i][2] /= 1.0 * FOLDS;
		}

		// 输出结果，选择最佳模型
		Classifier bestModel = null;
		double bestScore = -1;
		for (ListIterator<Classifier> it = models.listIterator(); it.hasNext();) {
			Classifier model = it.next();
			double fMeasure = measures[it.previousIndex()][2];
			System.out.println(model.getClass().getName() + "\n" + "\tRecall:    " + measures[it.previousIndex()][0]
					+ "\n" + "\tPrecision: " + measures[it.previousIndex()][1] + "\n" + "\tF-measure: " + fMeasure);
			if (fMeasure > bestScore) {
				bestScore = fMeasure;
				bestModel = model;
			}
		}
		System.out.println("Best model: " + bestModel.getClass().getName());

		// 可以继续做属性选择和特征生成，来应用更复杂的模型学习
	}

}
