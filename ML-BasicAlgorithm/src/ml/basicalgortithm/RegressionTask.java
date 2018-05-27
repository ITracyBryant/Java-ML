package ml.basicalgortithm;
/**
 * 
 */

/**
 * @XinCheng 2018年5月15日 Administrator
 * 使用weka实现回归模型，评估性能
 */

import java.io.File;
import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

public class RegressionTask {

	public static void main(String[] args) throws Exception {
		// 加载csv文件 （能源效率数据集）
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File("data/ENB2012_data.csv"));
		Instances data = loader.getDataSet();
		// System.out.println(data);

		// 8个属性描述建筑特征，目标变量为加热负载和冷却负载（后两个属性）
		// 构建回归模型
		// set class index to Y1(heating load) 在特征位置设置分类属性
		data.setClassIndex(data.numAttributes() - 2);
		// 移除最后属性Y2（冷却负载）
		Remove remove = new Remove();
		remove.setOptions(new String[] { "-R", data.numAttributes() + "" });
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);

		// bulid a regression model (经典线性回归模型)
		LinearRegression model = new LinearRegression();
		model.buildClassifier(data);
		System.out.println(model);

		// 10折交叉验证(10-fold cross-validation)
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(1), new Object[] {});
		System.out.println(eval.toSummaryString());
		// double coef[] = model.coefficients();
		System.out.println();

		// 构建线性回归树模型(Weka中M5类实现回归树--平滑线性模型)
		M5P md5 = new M5P();// 初始化模型
		md5.setOptions(new String[] { "" });// 传递参数数据
		md5.buildClassifier(data);
		System.out.println(md5);

		// 可视化回归树
		TreeVisualizer tv = new TreeVisualizer(null, md5.graph(), new PlaceNode2()); // 可视化
		JFrame frame = new javax.swing.JFrame("Regression Tree Visualizer");
		frame.setSize(1500, 600);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().add(tv);
		frame.setVisible(true);
		tv.fitToScreen();

		// 10-fold cross-validation
		eval.crossValidateModel(md5, data, 10, new Random(1), new Object[] {});
		System.out.println(eval.toSummaryString());
		System.out.println();

		// build additional models
		@SuppressWarnings("unused")
		ZeroR modelZero = new ZeroR();// 预测多数类，并被看作基准线
		REPTree modelTree = new REPTree();//
		modelTree.buildClassifier(data);
		System.out.println(modelTree);
		eval = new Evaluation(data);
		eval.crossValidateModel(modelTree, data, 10, new Random(1), new Object[] {});
		System.out.println(eval.toSummaryString());

		// SVM回归
		@SuppressWarnings("unused")
		SMOreg modelSVM = new SMOreg();
		@SuppressWarnings("unused")
		MultilayerPerceptron modelPerc = new MultilayerPerceptron(); // 多层感知机，神经网络分类器
		GaussianProcesses modelGP = new GaussianProcesses(); // 高斯过程回归
		modelGP.buildClassifier(data);
		System.out.println(modelGP);
		eval = new Evaluation(data);
		eval.crossValidateModel(modelGP, data, 10, new Random(1), new Object[] {});
		System.out.println(eval.toSummaryString());

		// Save ARFF
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		// saver.setFile(new File("data/ENB2012_data_1.arff"));
		saver.setDestination(new File("data/ENB2012_data.arff"));
		saver.writeBatch();

	}

}
