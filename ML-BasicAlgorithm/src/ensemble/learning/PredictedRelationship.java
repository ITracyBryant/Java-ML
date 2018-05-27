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
 * @XinCheng 2018年5月18日 Administrator http://kdd.org/kdd-cup/view/kdd-cup-2009
 *           通过集成方法预测客户关系
 */
public class PredictedRelationship {

	// static final enum
	static final int PREDICT_CHURN = 0, // 预测流失概率值
			PREDICT_APPETENCY = 1, // 预测购买概率值
			PREDICT_UPSELL = 3; // 预测追加销售概率值

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// 构建朴素贝叶斯基准线
		Classifier baselineNB = new NaiveBayes(); // 初始化朴素贝叶斯分类器
		double resNB[] = evaluate(baselineNB);
		System.out.println("Naive Bayes\n" + "\tchurn:     " + resNB[0] + "\n" + "\tappetency: " + resNB[1] + "\n"
				+ "\tup-sell:   " + resNB[2] + "\n" + "\toverall:   " + resNB[3] + "\n");

		// 创建集成模型库
		EnsembleLibrary ensembleLib = new EnsembleLibrary();
		// 添加决策树学习器
		ensembleLib.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2");
		ensembleLib.addModel("weka.classifiers.trees.J48 -S -C 0.25 -B -M 2 -A");
		// naive Bayes
		ensembleLib.addModel("weka.classifiers.bayes.NaiveBayes");
		// k-nn(基于lazy models)
		ensembleLib.addModel("weka.classifiers.lazy.IBk");
		// AdaBoost(集成方法)
		ensembleLib.addModel("weka.classifiers.meta.AdaBoostM1");
		// LogitBoost(基于逻辑回归的集成方法)
		ensembleLib.addModel("weka.classifiers.meta.LogitBoost");
		// SVM
		ensembleLib.addModel("weka.classifiers.functions.SMO");
		// Logistic regression
		ensembleLib.addModel("weka.classifiers.functions.Logistic");
		// Simple logistic regression
		ensembleLib.addModel("weka.classifiers.functions.SimpleLogistic");
		// Decision stump 决策树桩，基于单层决策树的集成方法
		ensembleLib.addModel("weka.classifiers.trees.DecisionStump");

		EnsembleLibrary.saveLibrary(new File("data/ensembleLib.model.xml"), ensembleLib, null);
		System.out.println(ensembleLib.getModels());

		// 初始化集成选择算法，指定优化ROC指标
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
																					// 模型库路径
				"-W", "data/esTmp", // </path/to/working/directory> - Specifies
									// the Working Directory, where all models
									// will be stored. 工作目录路径
				"-B", "10", // <numModelBags> - Set the number of bags, i.e.,
							// number of iterations to run the ensemble
							// selection algorithm. 模型包数
				"-E", "1.0", // <modelRatio> - Set the ratio of library models
								// that will be randomly chosen to populate each
								// bag of models. 模型比率
				"-V", "0.25", // <validationRatio> - Set the ratio of the
								// training data set that will be reserved for
								// validation. 验证比率
				"-H", "100", // <hillClimbIterations> - Set the number of
								// hillclimbing iterations to be performed on
								// each model bag. 爬山算法迭代次数
				"-I", "1.0", // <sortInitialization> - Set the the ratio of the
								// ensemble library that the sort initialization
								// algorithm will be able to choose from while
								// initializing the ensemble for each model bag.
								// 分类初始化
				"-X", "2", // <numFolds> - Sets the number of cross-validation
							// folds. 交叉验证的折数
				"-P", "roc", // <hillclimbMettric> - Specify the metric that
								// will be used for model selection during the
								// hillclimbing algorithm. 爬山算法用作模型选择的指标
				"-A", "forward", // <algorithm> - Specifies the algorithm to be
									// used for ensemble selection. 指定集成选择使用的算法
				"-R", "true", // - Flag whether or not models can be selected
								// more than once for an ensemble. 允许多次选择
				"-G", "true", // - Whether sort initialization greedily stops
								// adding models when performance degrades.
								// 性能下降时停止添加模型
				"-O", "true", // - Flag for verbose output. Prints out
								// performance of all selected models.
								// 详细输出所选中的模型性能信息
				"-S", "1", // <num> - Random number seed. 随机数种子
				"-D", "true" // - If set, classifier is run in debug mode and
								// may output additional info to the console.
								// 在调试模式下运行
		});

		// 性能评估，需要耗费计算机大量资源
		double resES[] = evaluate(ensembleSel);
		System.out.println("Ensemble Selection\n" + "\tchurn:     " + resES[0] + "\n" + "\tappetency: " + resES[1]
				+ "\n" + "\tup-sell:   " + resES[2] + "\n" + "\toverall:   " + resES[3] + "\n");

	}

	public static Instances loadData(String pathData, String pathLabels) throws Exception {
		// 加载数据
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator("\t");
		loader.setNominalAttributes("191-last");
		loader.setSource(new File(pathData));
		Instances data = loader.getDataSet();

		// 通过RemoveType过滤器将识别为String属性的空属性移走
		RemoveType removeString = new RemoveType();
		removeString.setOptions(new String[] { "-T", "string" }); // -T参数表示移走的特定类型的属性
		removeString.setInputFormat(data);
		Instances filteredData = Filter.useFilter(data, removeString);
		// 也可以使用Instances类中的deleteStringAttributes()进行移除操作，data.deleteStringAttributes();

		// 加载标签
		loader = new CSVLoader();
		loader.setFieldSeparator("/t");
		loader.setNoHeaderRowPresent(true); // 指定文件不带有任何标题行
		loader.setNominalAttributes("first-last");
		loader.setSource(new File(pathLabels));
		Instances labels = loader.getDataSet();
		// System.out.println(labels.toSummaryString());

		// 加载完了两个文件后，合并两个数据集，其中两个数据集的实例个数必须相同, 将标签添加为类值
		Instances labeledData = Instances.mergeInstances(filteredData, labels);
		// 将添加的标签属性用作目标变量，设置为类别
		labeledData.setClassIndex(labeledData.numAttributes() - 1);
		// System.out.println(labeledData.toSummaryString());
		return labeledData;
	}

	public static Instances preProcessData(Instances data) throws Exception {
		// 使用weka内置RemoveUseless过滤器移除无用属性
		RemoveUseless removeUseless = new RemoveUseless();
		removeUseless.setOptions(new String[] { "-M", "99" }); // 阈值，-M参数指定最大方差，只应用于nominal
																// attributes，默认值99%表明对于某个属性，如果超过99%实例拥有唯一的该属性值，则该属性会被移除
		removeUseless.setInputFormat(data);
		data = Filter.useFilter(data, removeUseless);

		// ReplaceMissingValues过滤器使用从训练数据得到的众数(nominal
		// attributes)与平均数(数值属性)，来代替数据集中的所有缺失值
		ReplaceMissingValues fixMissing = new ReplaceMissingValues();
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		// 对数值属性做离散化处理，使用Discretize过滤器将数值属性变换为区间(Intervals)
		Discretize discretizeNumeric = new Discretize();
		discretizeNumeric.setOptions(new String[] { "-B", "4", // -B参数，划分的区间数
				"-R", "first-last" // 指定属性范围(只有数值属性会被离散化)
		});
		fixMissing.setInputFormat(data);
		data = Filter.useFilter(data, fixMissing);

		// 只选择包含有用信息的属性
		// 检查每个属性包含的信息增益，使用AttributeSelection过滤器进行属性选择
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		// 从高于某个阈值的属性中只选择顶级属性，对高于特定阈值且带有信息增益的属性进行排列
		Ranker search = new Ranker();
		search.setOptions(new String[] { "-T", "0.001" }); // -T参数指定，保持低阈值(使属性至少带有一些信息)
		AttributeSelection attSelect = new AttributeSelection();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		// 应用属性选择
		attSelect.SelectAttributes(data);
		// 移除上次运行未被选中的属性
		data = attSelect.reduceDimensionality(data);
		return data;
	}

	public static double[] evaluate(Classifier model) throws Exception {
		double results[] = new double[4];
		String[] labelFiles = new String[] { "churn", "appetency", "upselling" };
		double overallScore = 0.0;
		for (int i = 0; i < labelFiles.length; i++) {
			// 加载数据
			Instances train_data = loadData("data/orange_small_train.data",
					"data/orange_small_train_" + labelFiles[i] + ".labels.txt");
			train_data = preProcessData(train_data);

			// 交叉验证集数据
			Evaluation eval = new Evaluation(train_data);
			eval.crossValidateModel(model, train_data, 5, new Random(1), new Object[] {}); // 5个折，验证基于数据的随机子集进行，传入一个随机种子

			// 保存结果集
			results[i] = eval.areaUnderROC(train_data.classAttribute().indexOfValue("1"));
			overallScore += results[i];
			System.out.println(labelFiles[i] + "\t-->\t" + results[i]);
		}
		// 取得关于三个标签问题的结果集的平均数
		results[3] = overallScore / 3;
		return results;
	}

}
