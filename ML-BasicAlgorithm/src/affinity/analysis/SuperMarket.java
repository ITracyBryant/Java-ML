/**
 * 
 */
package affinity.analysis;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.associations.Apriori;
import weka.associations.FPGrowth;
import weka.core.Instances;

/**
 * @XinCheng 2018��5��18�� Administrator �����������������Թ���ѧϰ��Apriori�㷨��FP(Ƶ��ģʽ)-�����㷨
 *           "������"--���������Ʒ--> ҽ����ϣ����������У��˿��ղ����ݣ��ͻ���ϵ����(CRM)��IT��Ӫ����
 *           ��ؼ��Ĳ������ڶԽ���ķ��������͹���(�����)
 */
public class SuperMarket {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// ��������
		Instances data = new Instances(new BufferedReader(new FileReader("data/supermarket.arff")));
		// ����ģ�ͣ���ʼ��Aprioriʵ�� ��Apriori�㷨���ϼ�С��С֧�ֶȣ�ֱ���ҵ�����Ĺ����������и�������С���Ŷ�
		Apriori model = new Apriori();
		model.buildAssociations(data); // ��ʼƵ��ģʽ�ھ�
		System.out.println(model);
		System.out.println("****************************");
		// ����FP-�����㷨ģ��
		FPGrowth fpgModel = new FPGrowth();
		fpgModel.buildAssociations(data);
		System.out.println(fpgModel);
		/* �ڴ���������ݼ�ʱ��ʹ��FP-�����㷨��Ҫ��ʱ�����Ը��̣�ִ��Ч�ʸ��� */
	}

}
