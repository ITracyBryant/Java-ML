/**
 * 
 */
package mallet.text.mining;

import java.io.File;
import java.util.ArrayList;
import java.util.regex.Pattern;

import cc.mallet.classify.Classifier;
import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.classify.Trial;
import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.FeatureSequence2FeatureVector;
import cc.mallet.pipe.Input2CharSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.pipe.iterator.FileIterator;
import cc.mallet.types.InstanceList;

/**
 * @XinCheng 2018年5月26日 Administrator 创建朴素贝叶斯垃圾邮件过滤器，使用词袋描述识别垃圾邮件
 *
 */
public class SpamDetector {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String stopListFilePath = "data/stoplists/en.txt";
		String dataFolderPath = "data/ex6DataEmails/train";
		String testFolderPath = "data/ex6DataEmails/test";

		// 创建一个默认流水线管道
		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
		pipeList.add(new Input2CharSequence("UTF-8"));
		Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");
		pipeList.add(new CharSequence2TokenSequence(tokenPattern));
		pipeList.add(new TokenSequenceLowercase());
		pipeList.add(new TokenSequenceRemoveStopwords(new File(stopListFilePath), "utf-8", false, false, false));
		pipeList.add(new TokenSequence2FeatureSequence());
		pipeList.add(new FeatureSequence2FeatureVector()); // 将特征序列转换为特征向量(特征向量中有数据时，可以使用任何一种分类算法)
		pipeList.add(new Target2Label());
		SerialPipes pipeline = new SerialPipes(pipeList);

		FileIterator folderIterator = new FileIterator(new File[] { new File(dataFolderPath) }, new TxtFilter(),
				FileIterator.LAST_DIRECTORY); // 加载训练样例，文件夹用作样例标签

		// 使用流水线新建实例列表，用于处理文本
		InstanceList instances = new InstanceList(pipeline);

		instances.addThruPipe(folderIterator); // 处理迭代器提供的每一个实例

		// Mallet实现了一组分类器
		@SuppressWarnings("rawtypes")
		ClassifierTrainer classifierTrainer = new NaiveBayesTrainer(); // 初始化朴素贝叶斯分类器
		Classifier classifier = classifierTrainer.train(instances); // 返回分类器

		// 在单独数据集上评价分类器之前，先导入test文件夹中的电子邮件
		InstanceList testInstances = new InstanceList(classifier.getInstancePipe());
		folderIterator = new FileIterator(new File[] { new File(testFolderPath) }, new TxtFilter(),
				FileIterator.LAST_DIRECTORY);
		testInstances.addThruPipe(folderIterator); // 通过训练期间初始化的流水线传递数据

		// 评估分类器性能，使用Trial类，并使用分类器与一系列测试样例对其初始化，初始化后立即执行评估
		Trial trial = new Trial(classifier, testInstances);

		System.out.println("Accuracy: " + trial.getAccuracy()); // 准确率
		System.out.println("F1 for class 'spam': " + trial.getF1("spam")); // F值(精确率和召回率的调和平均数)

		// 精确率
		System.out.println(
				"Precision for class '" + classifier.getLabelAlphabet().lookupLabel(1) + "': " + trial.getPrecision(1));

		// 召回率
		System.out.println(
				"Recall for class '" + classifier.getLabelAlphabet().lookupLabel(1) + "': " + trial.getRecall(1));

	}

}
