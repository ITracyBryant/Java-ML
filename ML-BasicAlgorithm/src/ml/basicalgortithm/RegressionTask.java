package ml.basicalgortithm;
/**
 * 
 */

/**
 * @XinCheng 2018��5��15�� Administrator
 * ʹ��wekaʵ�ֻع�ģ�ͣ���������
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
		// ����csv�ļ� ����ԴЧ�����ݼ���
		CSVLoader loader = new CSVLoader();
		loader.setFieldSeparator(",");
		loader.setSource(new File("data/ENB2012_data.csv"));
		Instances data = loader.getDataSet();
		// System.out.println(data);

		// 8��������������������Ŀ�����Ϊ���ȸ��غ���ȴ���أ����������ԣ�
		// �����ع�ģ��
		// set class index to Y1(heating load) ������λ�����÷�������
		data.setClassIndex(data.numAttributes() - 2);
		// �Ƴ��������Y2����ȴ���أ�
		Remove remove = new Remove();
		remove.setOptions(new String[] { "-R", data.numAttributes() + "" });
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);

		// bulid a regression model (�������Իع�ģ��)
		LinearRegression model = new LinearRegression();
		model.buildClassifier(data);
		System.out.println(model);

		// 10�۽�����֤(10-fold cross-validation)
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(1), new Object[] {});
		System.out.println(eval.toSummaryString());
		// double coef[] = model.coefficients();
		System.out.println();

		// �������Իع���ģ��(Weka��M5��ʵ�ֻع���--ƽ������ģ��)
		M5P md5 = new M5P();// ��ʼ��ģ��
		md5.setOptions(new String[] { "" });// ���ݲ�������
		md5.buildClassifier(data);
		System.out.println(md5);

		// ���ӻ��ع���
		TreeVisualizer tv = new TreeVisualizer(null, md5.graph(), new PlaceNode2()); // ���ӻ�
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
		ZeroR modelZero = new ZeroR();// Ԥ������࣬����������׼��
		REPTree modelTree = new REPTree();//
		modelTree.buildClassifier(data);
		System.out.println(modelTree);
		eval = new Evaluation(data);
		eval.crossValidateModel(modelTree, data, 10, new Random(1), new Object[] {});
		System.out.println(eval.toSummaryString());

		// SVM�ع�
		@SuppressWarnings("unused")
		SMOreg modelSVM = new SMOreg();
		@SuppressWarnings("unused")
		MultilayerPerceptron modelPerc = new MultilayerPerceptron(); // ����֪���������������
		GaussianProcesses modelGP = new GaussianProcesses(); // ��˹���̻ع�
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
