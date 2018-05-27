package ml.basicalgortithm;
/**
 * 
 */

/**
 * @XinCheng 2018��5��15�� Administrator
 * ͨ�������������ݼ�ʵ�ֻ���Weka�ľ���ģ��(������󻯾����㷨)ʶ�𳣼��ͻ��飬����������
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;

public class ClusteringTask {
	public static void main(String[] args) throws Exception {
		// �������ݣ����ݼ�����600��ʵ����ÿ��ʵ����11������������
		Instances data = new Instances(new BufferedReader(new FileReader("data/bank-data.arff")));

		// ����������ʵ�� ��EM(Expectation Maximization�������)�����㷨��
		// EM��ʹ�õĴ����������ֶ����ã�Ҳ��ͨ��������֤�Զ����ã���ʹ���ⲿ����(elbow-method)ȷ��--�÷�����鿴�ض����������͵�ƫ��ٷֱ�
		EM model = new EM();
		// ����������
		model.buildClusterer(data);
		System.out.println(model);

		// ʹ�ö�����Ȼ����(log likelihood measure)���������㷨����--������ʶ��Ĵص�һ�³̶�
		double logLikelihood = ClusterEvaluation.crossValidateModel(model, data, 10, new Random(1)); // ���ݼ����ֳ�10����(folds),���ÿ�������о���(�������㷨Ϊ�������ݸ����߸��ʣ������ڲ������ݽṹ����������ĸ���)
		System.out.println(logLikelihood);
	}

}
