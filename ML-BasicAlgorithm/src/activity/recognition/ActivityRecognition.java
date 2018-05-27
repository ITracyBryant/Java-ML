/**
 * 
 */
package activity.recognition;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 * @XinCheng 2018��5��26�� Administrator
 *
 */
public class ActivityRecognition {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		String databasePath = "data/features.arff";

		// ������������������ʾΪ�������ָ�ɵ������������Ϳ����ó���ලѧϰ�����㷨����������ѡ��������ɢ����ģ��ѧϰ��k�۽�����֤��
		// ��Ӧ���κ�һ��֧�������������㷨��SVM�����ɭ�֣�AdaBoost���������������磬����֪����
		// ����arff��ʽ�������ļ�
		Instances data = new Instances(new BufferedReader(new FileReader(databasePath)));

		// ����������ԣ�����class���������Ϊ���
		data.setClassIndex(data.numAttributes() - 1);

		// ��J48�ഴ��������ģ��
		String[] options = new String[] {};
		J48 model = new J48();
		model.setOptions(options);
		model.buildClassifier(data);

		// ���������
		System.out.println("Decision tree model:\n" + model);

		// ���ʵ�־�����ģ�͵�Դ�룬��ģ����Դ����ʽ����(Ҳ������weka��ʽ����ģ��)
		// ��һ��������Ƕ���ƶ�Ӧ��
		System.out.println("Source code:\n" + model.toSource("ActivityRecognitionEngine"));

		// ʹ��10�۽�����֤���ģ��׼ȷ�ȣ�������Щ������ʵ�������ƣ���10�۽�����֤�ж���������֣��п��ܻ���ѵ����������õ�ʵ��������ȫ��ͬ������֤����ܺ�
		// ����ʹ�ö�Ӧ�ڲ�ͬ���Ȼ�ͬ��Ⱥ���ۣ�����k�˽�����֤�����ÿ�����ظ������Խ����ƽ��
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(1), new Object[] {});
		System.out.println("Model performance:\n" + eval.toSummaryString());

		// ������Ϊһ����������Ϊ����(û���������ٸı�����һЩ�ӳ�)
		// ��������ʶ�����n��������Ϊ׷�ӵ�����������������ǿ��Ϊʶ�������������ܻ��û���ѧϰ�㷨��Ϊ��ǰ��Ϊ���Ǻ�ǰһ����ͬ������ʹ�����������������һ������n��������Ϊ����Щ������Ϊ����һ��������ʶ�𣬸÷������������κι�����Ϊ(��������һ��������Ӱ��)
		String[] activities = new String[] { "Walk", "Walk", "Walk", "Run", "Walk", "Run", "Run", "Sit", "Sit", "Sit" };
		DiscreteLowPass dlpFilter = new DiscreteLowPass(3); // ���ټ���ת����Ϊ��ȷ�����಻��̫�ױ䣬���һ����������������Ϊ�����п��ٸı����Ϊ
		for (String str : activities) {
			System.out.println(str + " -> " + dlpFilter.filter(str));
		}

	}

}

// ������Ϊ�б���window����
class DiscreteLowPass {

	List<Object> last;
	int window;

	public DiscreteLowPass(int window) {
		this.last = new ArrayList<Object>();
		this.window = window;
	}

	// ����һ������������ס��󴰿ڻ��������Ƶ����Ϊ���������Ϊӵ����ͬ�������򷵻������һ��
	public Object filter(Object obj) {
		if (last.size() < window) {
			last.add(obj); // ��û���㹻�۲�ֵ����洢����Ĺ۲�ֵ������
			return obj;
		}

		boolean same = true;
		for (Object o : last) {
			if (!o.equals(obj)) {
				same = false;
			}
		}
		if (same) {
			return obj;
		} else {
			Object o = getMostFrequentElement(last); // ������Ƶ���Ĺ۲�ֵ���Ƴ���ɵĹ۲�ֵ�������µĹ۲�ֵ
			last.add(obj);
			last.remove(0);
			return o;
		}
	}

	private Object getMostFrequentElement(List<Object> list) {

		HashMap<String, Integer> objectCounts = new HashMap<String, Integer>();
		Integer frequntCount = 0;
		Object frequentObject = null;

		for (Object obj : list) {
			String key = obj.toString();
			Integer count = objectCounts.get(key);
			if (count == null) {
				count = 0;
			}
			objectCounts.put(key, ++count);

			if (count >= frequntCount) {
				frequntCount = count;
				frequentObject = obj;
			}
		}

		return frequentObject; // ��hashmapʵ�ָ÷����������б�����Ƶ���Ķ���
	}

}
