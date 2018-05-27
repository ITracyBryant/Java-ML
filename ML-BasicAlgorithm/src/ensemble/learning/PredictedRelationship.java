/**
 * 
 */
package ensemble.learning;

import java.io.File;
import java.util.Random;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.EnsembleLibrary;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.EnsembleSelection;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * @XinCheng 2018��5��18�� Administrator http://kdd.org/kdd-cup/view/kdd-cup-2009
 *           ͨ�����ɷ���Ԥ��ͻ���ϵ
 */
public class PredictedRelationship {

	// static final enum
	static final int PREDICT_CHURN = 0, // Ԥ����ʧ����ֵ
			PREDICT_APPETENCY = 1, // Ԥ�⹺�����ֵ
			PREDICT_UPSELL = 3; // Ԥ��׷�����۸���ֵ

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// �������ر�Ҷ˹��׼��
		Classifier baselineNB = new NaiveBayes(); // ��ʼ�����ر�Ҷ˹������
		double resNB[] = evaluate(baselineNB);
		System.out.println("Naive Bayes\n" + "\tchurn:     " + resNB[0] + "\n" + "\tappetency: " + resNB[1] + "\n"
				+ "\tup-sell:   " + resNB[2] + "\n" + "\toverall:   " + resNB[3] + "\n");

		// ��������ģ�Ϳ�
		EnsembleLibrary ensembleLib = new EnsembleLibrary();
		// ��Ӿ�����ѧϰ��
		ensembleLib.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2");
		ensembleLib.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A");
		// naive Bayes
		ensembleLib.addModel("weka.classifiers.bayes.NaiveBayes");
		// k-nn(����lazy models)
		ensembleLib.addModel("weka.classifiers.lazy.IBk");
		// AdaBoost(���ɷ���)
		ensembleLib.addModel("weka.classifiers.meta.AdaBoostM1");
		// LogitBoost(�����߼��ع�ļ��ɷ���)
		ensembleLib.addModel("weka.classifiers.meta.LogitBoost");
		// SVM
		ensembleLib.addModel("weka.classifiers.functions.SMO");
		// Logistic regression
		ensembleLib.addModel("weka.classifiers.functions.Logistic");
		// Simple logistic regression
		ensembleLib.addModel("weka.classifiers.functions.SimpleLogistic");
		// Decision stump ������׮�����ڵ���������ļ��ɷ���
		ensembleLib.addModel("weka.classifiers.trees.DecisionStump");

		EnsembleLibrary.saveLibrary(new File("data/ensembleLib.model.xml"), ensembleLib, null);
		System.out.println(ensembleLib.getModels());

		// ��ʼ������ѡ���㷨��ָ���Ż�ROCָ��
		EnsembleSelection ensembleSel = new EnsembleSelection();
		ensembleSel.setOptions(new String[] { "-L", "data/ensembleLib.model.xml", // </path/to/modelLibrary>
																					// -
																					// Specifies
																					// the
																					// Model
																					// Library
																					// File,
																					// continuing
																					// the
																					// list
																					// of
																					// all
																					// models.
																					// ģ�Ϳ�·��
				"-W", "data/esTmp", // </path/to/working/directory> - Specifies
									// the Working Directory, where all models
									// will be stored. ����Ŀ¼·��
				"-B", "10", // <numModelBags> - Set the number of bags, i.e.,
							// number of iterations to run the ensemble
							// selection algorithm. ģ�Ͱ���
				"-E", "1.0", // <modelRatio> - Set the ratio of library models
								// that will be randomly chosen to populate each
								// bag of models. ģ�ͱ���
				"-V", "0.25", // <validationRatio> - Set the ratio of the
								// training data set that will be reserved for
								// validation. ��֤����
				"-H", "100", // <hillClimbIterations> - Set the number of
								// hillclimbing iterations to be performed on
								// each model bag. ��ɽ�㷨��������
				"-I", "1.0", // <sortInitialization> - Set the the ratio of the
								// ensemble library that the sort initialization
								// algorithm will be able to choose from while
								// initializing the ensemble for each model bag.
								// �����ʼ��
				"-X", "2", // <numFolds> - Sets the number of cross-validation
							// folds. ������֤������
				"-P", "roc", // <hillclimbMettric> - Specify the metric that
								// will be used for model selection during the
								// hillclimbing algorithm. ��ɽ�㷨����ģ��ѡ���ָ��
				"-A", "forward", // <algorithm> - Specifies the algorithm to be
									// used for ensemble selection. ָ������ѡ��ʹ�õ��㷨
				"-R", "true", // - Flag whether or not models can be selected
								// more than once for an ensemble. ������ѡ��
				"-G", "true", // - Whether sort initialization greedily stops
								// adding models when performance degrades.
								// �����½�ʱֹͣ���ģ��
				"-O", "true", // - Flag for verbose output. Prints out
								// performance of all selected models.
								// ��ϸ�����ѡ�е�ģ��������Ϣ
				"-S", "1", // <num> - Random number seed. ���������
				"-D", "true" // - If set, classifier is run in debug mode and
								// may output additional info to the console.
								// �ڵ���ģʽ������
		});

		// ������������Ҫ�ķѼ����������Դ
		double resES[] = evaluate(ensembleSel);
		System.out.println("Ensemble Selection\n" + "\tchurn:     " + resES[0] + "\n" + "\tappetency: " + resES[1]
				+ "\n" + "\tup-sell:   " + resES[2] + "\n" + "\toverall:   " + resES[3] + "\n");

	}

	public static Instances loadData(String pathData, String pathLabels) throws Exception {
		// ��������
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator("\t");
		loader.setNominalAttributes("191-last");
		loader.setSource(new File(pathData));
		Instances data = loader.getDataSet();

		// ͨ��RemoveType��������ʶ��ΪString���ԵĿ���������
		RemoveType removeString = new RemoveType();
		removeString.setOptions(new String[] { "-T", "string" }); // -T������ʾ���ߵ��ض����͵�����
		removeString.setInputFormat(data);
		Instances filteredData = Filter.useFilter(data, removeString);
		// Ҳ����ʹ��Instances���е�deleteStringAttributes()�����Ƴ�������data.deleteStringAttributes();

		// ���ر�ǩ
		loader = new CSVLoader();
		loader.setFieldSeparator("/t");
		loader.setNoHeaderRowPresent(true); // ָ���ļ��������κα�����
		loader.setNominalAttributes("first-last");
		loader.setSource(new File(pathLabels));
		Instances labels = loader.getDataSet();
		// System.out.println(labels.toSummaryString());

		// �������������ļ��󣬺ϲ��������ݼ��������������ݼ���ʵ������������ͬ, ����ǩ���Ϊ��ֵ
		Instances labeledData = Instances.mergeInstances(filteredData, labels);
		// ����ӵı�ǩ��������Ŀ�����������Ϊ���
		labeledData.setClassIndex(labeledData.numAttributes() - 1);
		// System.out.println(labeledData.toSummaryString());
		return labeledData;
	}

	public static Instances preProcessData(Instances data) throws Exception {
		// ʹ��weka����RemoveUseless�������Ƴ���������
		RemoveUseless removeUseless = new RemoveUseless();
		removeUseless.setOptions(new String[] { "-M", "99" }); // ��ֵ��-M����ָ����󷽲ֻӦ����nominal
																// attributes��Ĭ��ֵ99%��������ĳ�����ԣ��������99%ʵ��ӵ��Ψһ�ĸ�����ֵ��������Իᱻ�Ƴ�
		removeUseless.setInputFormat(data);
		data = Filter.useFilter(data, removeUseless);

		// ReplaceMissingValues������ʹ�ô�ѵ�����ݵõ�������(nominal
		// attributes)��ƽ����(��ֵ����)�����������ݼ��е�����ȱʧֵ
		ReplaceMissingValues fixMissing = new ReplaceMissingValues();
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		// ����ֵ��������ɢ������ʹ��Discretize����������ֵ���Ա任Ϊ����(Intervals)
		Discretize discretizeNumeric = new Discretize();
		discretizeNumeric.setOptions(new String[] { "-B", "4", // -B���������ֵ�������
				"-R", "first-last" // ָ�����Է�Χ(ֻ����ֵ���Իᱻ��ɢ��)
		});
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		// ֻѡ�����������Ϣ������
		// ���ÿ�����԰�������Ϣ���棬ʹ��AttributeSelection��������������ѡ��
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		// �Ӹ���ĳ����ֵ��������ֻѡ�񶥼����ԣ��Ը����ض���ֵ�Ҵ�����Ϣ��������Խ�������
		Ranker search = new Ranker();
		search.setOptions(new String[] { "-T", "0.001" }); // -T����ָ�������ֵ���ֵ(ʹ�������ٴ���һЩ��Ϣ)
		AttributeSelection attSelect = new AttributeSelection();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		// Ӧ������ѡ��
		attSelect.SelectAttributes(data);
		// �Ƴ��ϴ�����δ��ѡ�е�����
		data = attSelect.reduceDimensionality(data);
		return data;
	}

	public static double[] evaluate(Classifier model) throws Exception {
		double results[] = new double[4];
		String[] labelFiles = new String[] { "churn", "appetency", "upselling" };
		double overallScore = 0.0;
		for (int i = 0; i < labelFiles.length; i++) {
			// ��������
			Instances train_data = loadData("data/orange_small_train.data",
					"data/orange_small_train_" + labelFiles[i] + ".labels.txt");
			train_data = preProcessData(train_data);

			// ������֤������
			Evaluation eval = new Evaluation(train_data);
			eval.crossValidateModel(model, train_data, 5, new Random(1), new Object[] {}); // 5���ۣ���֤�������ݵ�����Ӽ����У�����һ���������

			// ��������
			results[i] = eval.areaUnderROC(train_data.classAttribute().indexOfValue("1"));
			overallScore += results[i];
			System.out.println(labelFiles[i] + "\t-->\t" + results[i]);
		}
		// ȡ�ù���������ǩ����Ľ������ƽ����
		results[3] = overallScore / 3;
		return results;
	}

}
