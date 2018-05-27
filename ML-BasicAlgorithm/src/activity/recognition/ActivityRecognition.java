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
 * @XinCheng 2018年5月26日 Administrator
 *
 */
public class ActivityRecognition {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		String databasePath = "data/features.arff";

		// 当传感器数据样本表示为带有类别指派的特征向量，就可以用常规监督学习分类算法，包括特征选择，特征离散化，模型学习，k折交叉验证等
		// 可应用任何一种支持数字特征的算法，SVM，随机森林，AdaBoost，决策树，神经网络，多层感知器等
		// 加载arff格式的数据文件
		Instances data = new Instances(new BufferedReader(new FileReader(databasePath)));

		// 创建类别属性，设置class的最后属性为类别
		data.setClassIndex(data.numAttributes() - 1);

		// 用J48类创建决策树模型
		String[] options = new String[] {};
		J48 model = new J48();
		model.setOptions(options);
		model.buildClassifier(data);

		// 输出决策树
		System.out.println("Decision tree model:\n" + model);

		// 输出实现决策树模型的源码，将模型以源码形式导出(也可以以weka格式导出模型)
		// 将一个分类器嵌入移动应用
		System.out.println("Source code:\n" + model.toSource("ActivityRecognitionEngine"));

		// 使用10折交叉验证检测模型准确度，由于这些连续的实例很类似，在10折交叉验证中对其随机划分，有可能会让训练与测试所用的实例几乎完全相同，故验证结果很好
		// 可以使用对应于不同组测度或不同人群的折，运行k人交叉验证，针对每个人重复，并对结果做平均
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(model, data, 10, new Random(1), new Object[] {});
		System.out.println("Model performance:\n" + eval.toSummaryString());

		// 输出结果为一个连续的行为序列(没有做出快速改变会带来一些延迟)
		// 将分类器识别过的n个过往行为追加到特征向量，可以增强行为识别能力，但可能会让机器学习算法认为当前行为总是和前一个相同。可以使用两个分类器解决，一个包含n个过往行为，这些过往行为由另一个分类器识别，该分类器不包含任何过往行为(不会受另一分类器的影响)
		String[] activities = new String[] { "Walk", "Walk", "Walk", "Run", "Walk", "Run", "Run", "Sit", "Sit", "Sit" };
		DiscreteLowPass dlpFilter = new DiscreteLowPass(3); // 减少假性转换，为了确保分类不会太易变，设计一个过滤器，过滤行为序列中快速改变的行为
		for (String str : activities) {
			System.out.println(str + " -> " + dlpFilter.filter(str));
		}

	}

}

// 包含行为列表与window参数
class DiscreteLowPass {

	List<Object> last;
	int window;

	public DiscreteLowPass(int window) {
		this.last = new ArrayList<Object>();
		this.window = window;
	}

	// 创建一个过滤器，记住最后窗口活动，返回最频繁行为，若多个行为拥有相同分数，则返回最近的一个
	public Object filter(Object obj) {
		if (last.size() < window) {
			last.add(obj); // 若没有足够观测值，会存储传入的观测值并返回
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
			Object o = getMostFrequentElement(last); // 返回最频繁的观测值，移除最旧的观测值，插入新的观测值
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

		return frequentObject; // 用hashmap实现该方法，返回列表中最频繁的对象
	}

}
