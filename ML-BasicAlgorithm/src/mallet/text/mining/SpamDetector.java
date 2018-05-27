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
 * @XinCheng 2018��5��26�� Administrator �������ر�Ҷ˹�����ʼ���������ʹ�ôʴ�����ʶ�������ʼ�
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

		// ����һ��Ĭ����ˮ�߹ܵ�
		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
		pipeList.add(new Input2CharSequence("UTF-8"));
		Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+");
		pipeList.add(new CharSequence2TokenSequence(tokenPattern));
		pipeList.add(new TokenSequenceLowercase());
		pipeList.add(new TokenSequenceRemoveStopwords(new File(stopListFilePath), "utf-8", false, false, false));
		pipeList.add(new TokenSequence2FeatureSequence());
		pipeList.add(new FeatureSequence2FeatureVector()); // ����������ת��Ϊ��������(����������������ʱ������ʹ���κ�һ�ַ����㷨)
		pipeList.add(new Target2Label());
		SerialPipes pipeline = new SerialPipes(pipeList);

		FileIterator folderIterator = new FileIterator(new File[] { new File(dataFolderPath) }, new TxtFilter(),
				FileIterator.LAST_DIRECTORY); // ����ѵ���������ļ�������������ǩ

		// ʹ����ˮ���½�ʵ���б����ڴ����ı�
		InstanceList instances = new InstanceList(pipeline);

		instances.addThruPipe(folderIterator); // ����������ṩ��ÿһ��ʵ��

		// Malletʵ����һ�������
		@SuppressWarnings("rawtypes")
		ClassifierTrainer classifierTrainer = new NaiveBayesTrainer(); // ��ʼ�����ر�Ҷ˹������
		Classifier classifier = classifierTrainer.train(instances); // ���ط�����

		// �ڵ������ݼ������۷�����֮ǰ���ȵ���test�ļ����еĵ����ʼ�
		InstanceList testInstances = new InstanceList(classifier.getInstancePipe());
		folderIterator = new FileIterator(new File[] { new File(testFolderPath) }, new TxtFilter(),
				FileIterator.LAST_DIRECTORY);
		testInstances.addThruPipe(folderIterator); // ͨ��ѵ���ڼ��ʼ������ˮ�ߴ�������

		// �������������ܣ�ʹ��Trial�࣬��ʹ�÷�������һϵ�в������������ʼ������ʼ��������ִ������
		Trial trial = new Trial(classifier, testInstances);

		System.out.println("Accuracy: " + trial.getAccuracy()); // ׼ȷ��
		System.out.println("F1 for class 'spam': " + trial.getF1("spam")); // Fֵ(��ȷ�ʺ��ٻ��ʵĵ���ƽ����)

		// ��ȷ��
		System.out.println(
				"Precision for class '" + classifier.getLabelAlphabet().lookupLabel(1) + "': " + trial.getPrecision(1));

		// �ٻ���
		System.out.println(
				"Recall for class '" + classifier.getLabelAlphabet().lookupLabel(1) + "': " + trial.getRecall(1));

	}

}
