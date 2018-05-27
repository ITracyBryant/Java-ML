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
 * @XinCheng 2018��5��22�� Administrator Ϊ����ģʽ��ģ������������թ���
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

		// �������ݼ�
		// ��������
		int CLASS_INDEX = 15;
		int POLICY_INDEX = 17;
		@SuppressWarnings("unused")
		int NO_FRAUD = 0, FRAUD = 1;
		int FOLDS = 3;

		// ȷ�������������Ͷ���������
		NumericToNominal toNominal = new NumericToNominal();
		toNominal.setInputFormat(data);
		data = Filter.useFilter(data, toNominal);

		// ָ����Ԥ������ԣ����ö�Ӧ��������
		data.setClassIndex(CLASS_INDEX);

		// �Ƴ�һ������������ŵ�����(policy attribute)
		Remove remove = new Remove();
		remove.setInputFormat(data);
		remove.setOptions(new String[] { "-R", "" + POLICY_INDEX });
		data = Filter.useFilter(data, remove);

		System.out.println(data.toSummaryString());
		System.out.println("�����������ԣ�\n" + data.attributeStats(data.classIndex()));

		// Vanilla approach ������--�Ȳ���Ҫ������Ԥ����Ҳ����Ҫ�������ݼ�����ϸ�ڣ���k�۽�����֤
		// ���������
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

		// ���ݼ�����(Dataset
		// rebalancing),�ֹ�ʵ��k�۽�����֤��Weka���ù�����Resample���ڴ�һ�����ݼ��������ȡ��������ʹ�����ó��������ó��������Խ�һ���ֲ�����Ϊ����ȷֲ�
		// ʹ��StratifiedRemoveFolds�������������ݼ������ֺ�ĸ�������Ȼ������ͬ����ֲ�
		StratifiedRemoveFolds kFold = new StratifiedRemoveFolds();
		kFold.setInputFormat(data);
		double measures[][] = new double[models.size()][FOLDS];
		for (int k = 1; k <= FOLDS; k++) {
			// �����ݻ��ֳɲ����ۺ�ѵ����
			kFold.setOptions(new String[] { "-N", "" + FOLDS, "-F", "" + k, "-S", "1" });
			Instances test = Filter.useFilter(data, kFold);
			kFold.setOptions(new String[] { "-N", "" + FOLDS, "-F", "" + k, "-S", "1", "-V" });// ��ѡ
																								// ����"-V"
			Instances train = Filter.useFilter(data, kFold);
			System.out.println("Fold" + k + "\n\ttrain: " + train.size() + "\n\ttest: " + test.size());

			// ����ѵ������-Z����ָ��Ҫ�س������ݼ��ı�����-B��������ֲ�����Ϊ���ȷֲ�
			Resample resample = new Resample();
			resample.setInputFormat(data);
			resample.setOptions(new String[] { "-Z", "100", "-B", "1" }); // �����滻
			Instances balancedTrain = Filter.useFilter(train, resample);

			// ������������������
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

				// Ϊ����ƽ����������
				measures[it.previousIndex()][0] += eval.recall(FRAUD);
				measures[it.previousIndex()][1] += eval.precision(FRAUD);
				measures[it.previousIndex()][2] += eval.fMeasure(FRAUD);
			}
		}

		// ����ƽ����
		for (int i = 0; i < models.size(); i++) {
			measures[i][0] /= 1.0 * FOLDS;
			measures[i][1] /= 1.0 * FOLDS;
			measures[i][2] /= 1.0 * FOLDS;
		}

		// ��������ѡ�����ģ��
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

		// ���Լ���������ѡ����������ɣ���Ӧ�ø����ӵ�ģ��ѧϰ
	}

}
