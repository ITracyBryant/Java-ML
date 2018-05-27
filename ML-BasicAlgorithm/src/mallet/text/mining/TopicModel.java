/**
 * 
 */
package mallet.text.mining;

import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Formatter;
import java.util.Iterator;
import java.util.Locale;
import java.util.TreeSet;
import java.util.regex.Pattern;

import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.Input2CharSequence;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.TokenSequenceLowercase;
import cc.mallet.pipe.TokenSequenceRemoveStopwords;
import cc.mallet.pipe.iterator.FileIterator;
import cc.mallet.topics.MarginalProbEstimator;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.IDSorter;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.util.Randoms;

/**
 * @XinCheng 2018��5��26�� Administrator
 *           ����ģ����һ���޼ලѧϰ������ģ�ʹ��ı����Ͽ���Ѱ��ģʽ������һ����ͳ������ķ�ʽʶ�����⣬��ɵ��ʱ�
 *           �����������׷ֲ��㷨(Latent Dirichlet Allocation)
 *           �������ߴӿ��ܵĵ�������ѡ�񵥴����һ�����֣�ÿ�����Ӷ�Ӧһ�����⣬���㷨���Դ���ѧ�ϰ��ı���⵽��Ӧ�����ӣ�
 *           ��Щ�����ǲ��󵥴����п��ܵ���Դ��Ȼ�󲻶ϵ����ù��̣�ֱ�������дʷ��䵽���п��ܵ����ӣ���Щ���Ӿͳ�Ϊ����
 *           �ı��ھ򣺽�û�нṹ����Ȼ����ת��Ϊ�ṹ���Ļ������Ե�ʵ��
 *           �ı���ȡ-->�ı��з�-->����淶��-->�ϴ�(Tokenization)-->�Ƴ�ֹͣ��-->POS(���Ա�ע)�����λ�ԭ-->
 *           �����ת��Ϊ�����ռ�--�ʴ�(BoW)-->ʵ��
 */
public class TopicModel {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		String dataFolderPath = "data/bbc"; // ΪBBC���Ź�������ģ��
		String stopListFilePath = "data/stoplists/en.txt";

		// ���������������Ϊδ������ĵ�ָ��һ������
		// ����һ��Ĭ�Ϲܵ���ÿ���������Ӧ��Mallet�е�һ���ܵ����ô��з�ʽ�ѹܵ���������
		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
		pipeList.add(new Input2CharSequence("UTF-8"));
		Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+"); // ��������ʽ��ԭʼ�ַ������б�ǻ�����ģʽ����Unicode�ַ������֣��»���
		pipeList.add(new CharSequence2TokenSequence(tokenPattern));
		pipeList.add(new TokenSequenceLowercase()); // �������ַ�ת��ΪСд
		pipeList.add(new TokenSequenceRemoveStopwords(new File(stopListFilePath), "utf-8", false, false, false)); // ʹ�ñ�׼Ӣ��ֹͣ�ʱ�����ֹͣ��(������Ԥ�������ĸ�Ƶ��)������ʱ�Ƿ����ִ�Сд��ɾ�����Ƿ���
		pipeList.add(new TokenSequence2FeatureSequence()); // ������ʵ�ʵ��ʣ�������ת������������ʾ�����ڴʴ��е�����
		pipeList.add(new Target2Label()); // �����ǩ���ñ�ǩ�ַ���ʹ��һ����������ʾ��ǩ�ڴʴ��е�λ��
		SerialPipes pipeline = new SerialPipes(pipeList); // ��󽫹ܵ��б�洢��SerialPipes�࣬����ͨ��һϵ�йܵ�ת��ʵ��

		FileIterator folderIterator = new FileIterator(new File[] { new File(dataFolderPath) }, new TxtFilter(),
				FileIterator.LAST_DIRECTORY); // ��·����ȡ�ļ�����ʼ��folderIterator����һ������ָ�����ļ���·�����ڶ���������������������.txt�ļ��ϣ����һ�������÷�����·���е����Ŀ¼����Ϊ���ǩ

		// �½�ʵ���б�����Ҫ���ڴ����ı��Ĺܵ����ݸ���
		InstanceList instances = new InstanceList(pipeline);

		// ���������������ÿ��ʵ��
		instances.addThruPipe(folderIterator);

		// alpha_t = 0.01, beta_w = 0.01
		// ��alphaֵ��ʾÿ���ĵ����ܻ�϶�����⣬����ָĳһ���⣻��alphaֵʹ�ĵ���������Щ����Լ������ζ���ĵ����ܻ�ϼ������⣬Ҳ����ֻ��һ������
		// ��betaֵ��ʾÿ��������ܻ�Ϻܶ൥�ʣ������ض�ĳ�����ʣ���betaֵ��ʾ����ֻ��ϼ�������
		// ʹ��ParallelTopicModel��ʵ��һ���������׷ֲ�ģ�ͣ���������5�������ģ��
		int numTopics = 5;
		ParallelTopicModel model = new ParallelTopicModel(numTopics, 0.01, 0.01);

		model.addInstances(instances);

		// ʹ�ò���ʵ�֣�ָ������ִ�е��߳���
		model.setNumThreads(4);

		// ���õ�����������ģ��
		model.setNumIterations(50);
		model.estimate(); // ʵ�ʴ���һ��LDAģ��

		// LL/tokenָģ�͵Ķ������ƶ�(log-likelihood)���Ա����������ʾ���������ģ�͵����Ƴ̶ȣ���ֵԽ�󣬱�ʾģ��Ʒ��Խ��
		/*
		 * Saving model
		 */

		// ��ģ��Ӧ�õ������ݣ�Ҫ��ģ��һ�𱣴�/���عܵ���ʹ�øùܵ������ʵ��
		String modelPath = "myTopicModel";

		// ���л��������
		// ����ģ��
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(modelPath + ".model")));
		oos.writeObject(model);
		oos.close();
		// ������ˮ��
		oos = new ObjectOutputStream(new FileOutputStream(new File(modelPath + ".pipeline")));
		oos.writeObject(pipeline);
		oos.close();

		System.out.println("Model saved.");

		/*
		 * Loading the model
		 */
		// ��ObjectInputStream��ָ�ģ��
		// ����ģ��
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(modelPath + ".model")));
		model = (ParallelTopicModel) ois.readObject();
		ois.close();
		// ������ˮ��
		ois = new ObjectInputStream(new FileInputStream(new File(modelPath + ".pipeline")));
		pipeline = (SerialPipes) ois.readObject();
		ois.close();

		System.out.println("Model loaded.");

		// Show the words and topics in the first instance

		// The data alphabet maps word IDs to strings
		Alphabet dataAlphabet = instances.getDataAlphabet();

		FeatureSequence tokens = (FeatureSequence) model.getData().get(0).instance.getData();
		LabelSequence topics = model.getData().get(0).topicSequence;

		Formatter out = new Formatter(new StringBuilder(), Locale.US);
		for (int position = 0; position < tokens.getLength(); position++) {
			out.format("%s-%d ", dataAlphabet.lookupObject(tokens.getIndexAtPosition(position)),
					topics.getIndexAtPosition(position));
		}
		System.out.println(out);

		// Estimate the topic distribution of the first instance,
		// given the current Gibbs state.
		double[] topicDistribution = model.getTopicProbabilities(0);

		// Get an array of sorted sets of word ID/count pairs
		ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();

		// Show top 5 words in topics with proportions for the first document
		for (int topic = 0; topic < numTopics; topic++) {
			Iterator<IDSorter> iterator = topicSortedWords.get(topic).iterator();

			out = new Formatter(new StringBuilder(), Locale.US);
			out.format("%d\t%.3f\t", topic, topicDistribution[topic]);
			int rank = 0;
			while (iterator.hasNext() && rank < 5) {
				IDSorter idCountPair = iterator.next();
				out.format("%s (%.0f) ", dataAlphabet.lookupObject(idCountPair.getID()), idCountPair.getWeight());
				rank++;
			}
			System.out.println(out);
		}

		/*
		 * Testing
		 */

		// Wallach�����һ�ֺ���ģ�������ķ������÷����������ģ���³�ȡ�ĵ��Ķ������ʣ���δ�������ĵ��Ŀ����������Ƚ�ģ��-->������Խ�ߣ�ģ��Խ��
		System.out.println("Evaluation");

		// �������ݼ�
		InstanceList[] instanceSplit = instances.split(new Randoms(), new double[] { 0.9, 0.1, 0.0 });

		// ʹ��ǰ90%����ѵ��
		model.addInstances(instanceSplit[0]);
		model.setNumThreads(4);
		model.setNumIterations(50);
		model.estimate();

		// ��ȡ��������ʵ���������ĵ���Wallach��������
		MarginalProbEstimator estimator = model.getProbEstimator();
		double loglike = estimator.evaluateLeftToRight(instanceSplit[1], 10, false, null);
		System.out.println("Total log likelihood: " + loglike);

	}

}

/** This class illustrates how to build a simple file filter */
class TxtFilter implements FileFilter {

	/**
	 * Test whether the string representation of the file ends with the correct
	 * extension. Note that {@ref FileIterator} will only call this filter if
	 * the file is not a directory, so we do not need to test that it is a file.
	 */
	public boolean accept(File file) {
		return file.toString().endsWith(".txt");

	}

}
