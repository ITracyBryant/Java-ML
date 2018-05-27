package ml.basicalgortithm;
/**
 * 
 */

/**
 * @XinCheng 2018年5月15日 Administrator
 *
 * 使用Weka中的J48类，实现了C4.5决策树学习器
 */

import java.util.Random;

import javax.swing.JFrame;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

public class ClassificationTask {
	@SuppressWarnings({ "unused", "static-access" })
	public static void main(String[] args) throws Exception {
		// 加载数据，所有数据样本都包含在一个Instances对象中
		DataSource source = new DataSource("data/zoo.arff");
		Instances data = source.getDataSet();
		System.out.println(data.numInstances() + " instances loaded.");
		// System.out.println(data.toString());

		// 过滤需要预测新样本的属性值
		String[] opts = new String[] { "-R", "1" }; // 设置一个参数的字符串表，指定必须移走第一个属性
		Remove remove = new Remove();
		remove.setOptions(opts);
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove); // 将过滤器应用于所选数据集

		// 特征选择(属性选择)
		InfoGainAttributeEval eval = new InfoGainAttributeEval(); // 将信息增益用作评价器
		Ranker search = new Ranker(); // 通过信息增益分数对特征进行分类排序
		AttributeSelection attSelect = new AttributeSelection();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		attSelect.SelectAttributes(data);
		int[] indices = attSelect.selectedAttributes();
		System.out.println("Selected attributes: " + Utils.arrayToString(indices));

		// 用J48构建决策树
		String[] options = new String[1]; // 使用字符串表传递额外参数U，创建一颗J48未剪枝树
		options[0] = "-U";
		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(data); // 对学习过程初始化
		System.out.println(tree);

		// 利用内建的TreeVisualizer树浏览器可视化未剪枝树
		TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2()); // 可视化
		JFrame frame = new javax.swing.JFrame("Tree Visualizer");
		frame.setSize(800, 600);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(tv);
		frame.setVisible(true);
		tv.fitToScreen();

		/*
		 * 创建分类器步骤： 1.初始化一个分类器 2.传递控制模型复杂度的参数 3。调用buildClassifier(Instances)
		 */

		// Classify new instance
		// 构造一个特征向量描述新样本
		double[] vals = new double[data.numAttributes()];
		vals[0] = 1.0; // hair {false, true}
		vals[1] = 0.0; // feathers {false, true}
		vals[2] = 0.0; // eggs {false, true}
		vals[3] = 1.0; // milk {false, true}
		vals[4] = 0.0; // airborne {false, true}
		vals[5] = 0.0; // aquatic {false, true}
		vals[6] = 0.0; // predator {false, true}
		vals[7] = 1.0; // toothed {false, true}
		vals[8] = 1.0; // backbone {false, true}
		vals[9] = 1.0; // breathes {false, true}
		vals[10] = 1.0; // venomous {false, true}
		vals[11] = 0.0; // fins {false, true}
		vals[12] = 4.0; // legs INTEGER [0,9]
		vals[13] = 1.0; // tail {false, true}
		vals[14] = 1.0; // domestic {false, true}
		vals[15] = 0.0; // catsize {false, true}
		Instance myUnicorn = new DenseInstance(1.0, vals);
		myUnicorn.setDataset(data);

		double label = tree.classifyInstance(myUnicorn);
		System.out.println(data.classAttribute().value((int) label)); // 最后输出mammal类标签

		// CrossValidation评估模型性能
		Classifier cl = new J48();
		Evaluation eval_roc = new Evaluation(data); // 评估ROC曲线
		eval_roc.crossValidateModel(cl, data, 10, new Random(1), new Object[] {}); // 提供模型，数据，折数，初始的随机种子
		System.out.println(eval_roc.toSummaryString());
		// 通过检查混淆矩阵查看特定的错误分类出现在何处, 混淆矩阵可以直观的看到分类模型所犯错误的具体类型
		double[][] confusionMatrix = eval_roc.confusionMatrix();
		System.out.println(eval_roc.toMatrixString());

		// 绘制ROC曲线，查看模型评估结果
		ThresholdCurve tc = new ThresholdCurve();
		int classIndex = 0;
		Instances result = tc.getCurve(eval_roc.predictions(), classIndex);
		// plot curve
		ThresholdVisualizePanel tvp = new ThresholdVisualizePanel();
		tvp.setROCString("(Area under ROC = " + tc.getROCArea(result) + ")");
		tvp.setName(result.relationName());
		PlotData2D tempd = new PlotData2D(result);
		tempd.setPlotName(result.relationName());
		tempd.addInstanceNumberAttribute();
		// 绘制连接特定的数据点
		boolean[] cp = new boolean[result.numInstances()];
		for (int n = 1; n < cp.length; n++)
			cp[n] = true;
		tempd.setConnectPoints(cp);

		// add plot
		tvp.addPlot(tempd);
		// display curve
		JFrame frameRoc = new javax.swing.JFrame("ROC Curve");
		frameRoc.setSize(800, 600);
		frameRoc.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frameRoc.getContentPane().add(tvp);
		frameRoc.setVisible(true);
	}

}
