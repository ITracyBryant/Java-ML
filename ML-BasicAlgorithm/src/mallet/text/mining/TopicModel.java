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
 * @XinCheng 2018年5月26日 Administrator
 *           主题模型是一种无监督学习，主题模型从文本语料库中寻找模式，采用一种有统计意义的方式识别主题，组成单词表
 *           隐含狄利克雷分布算法(Latent Dirichlet Allocation)
 *           假设作者从可能的单词篮中选择单词组成一段文字，每个篮子对应一个主题，该算法可以从数学上把文本拆解到相应的篮子，
 *           这些篮子是拆解后单词最有可能的来源，然后不断迭代该过程，直到将所有词分配到最有可能的篮子，这些篮子就称为主题
 *           文本挖掘：将没有结构的自然语言转换为结构化的基于属性的实例
 *           文本提取-->文本切分-->编码规范化-->断词(Tokenization)-->移除停止词-->POS(词性标注)，词形还原-->
 *           将标记转换为特征空间--词袋(BoW)-->实例
 */
public class TopicModel {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		String dataFolderPath = "data/bbc"; // 为BBC新闻构建主题模型
		String stopListFilePath = "data/stoplists/en.txt";

		// 创建主题分类器，为未分类的文档指派一个标题
		// 创建一个默认管道，每个处理步骤对应于Mallet中的一个管道，用串行方式把管道连接起来
		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
		pipeList.add(new Input2CharSequence("UTF-8"));
		Pattern tokenPattern = Pattern.compile("[\\p{L}\\p{N}_]+"); // 用正则表达式对原始字符串进行标记化，该模式包含Unicode字符，数字，下划线
		pipeList.add(new CharSequence2TokenSequence(tokenPattern));
		pipeList.add(new TokenSequenceLowercase()); // 将所有字符转换为小写
		pipeList.add(new TokenSequenceRemoveStopwords(new File(stopListFilePath), "utf-8", false, false, false)); // 使用标准英文停止词表，移走停止词(不具有预测能力的高频词)，移走时是否区分大小写，删除后是否标记
		pipeList.add(new TokenSequence2FeatureSequence()); // 不保存实际单词，将它们转换成整数，表示单词在词袋中的索引
		pipeList.add(new Target2Label()); // 对类标签不用标签字符串使用一个整数，表示标签在词袋中的位置
		SerialPipes pipeline = new SerialPipes(pipeList); // 最后将管道列表存储到SerialPipes类，该类通过一系列管道转换实例

		FileIterator folderIterator = new FileIterator(new File[] { new File(dataFolderPath) }, new TxtFilter(),
				FileIterator.LAST_DIRECTORY); // 从路径读取文件，初始化folderIterator，第一个参数指定根文件夹路径，第二个参数将迭代器限制在.txt文件上，最后一个参数让方法将路径中的最后目录名作为类标签

		// 新建实例列表，将需要用于处理文本的管道传递给它
		InstanceList instances = new InstanceList(pipeline);

		// 处理迭代器给出的每个实例
		instances.addThruPipe(folderIterator);

		// alpha_t = 0.01, beta_w = 0.01
		// 高alpha值表示每个文档可能混合多个主题，不特指某一主题；低alpha值使文档较少受这些条件约束，意味着文档可能混合几个主题，也可能只有一个主题
		// 高beta值表示每个主题可能混合很多单词，不是特定某个单词；低beta值表示主题只混合几个单词
		// 使用ParallelTopicModel类实现一个狄利克雷分布模型，创建带有5个主题的模型
		int numTopics = 5;
		ParallelTopicModel model = new ParallelTopicModel(numTopics, 0.01, 0.01);

		model.addInstances(instances);

		// 使用并行实现，指定并行执行的线程数
		model.setNumThreads(4);

		// 设置迭代次数运行模型
		model.setNumIterations(50);
		model.estimate(); // 实际创建一个LDA模型

		// LL/token指模型的对数相似度(log-likelihood)除以标记总数，表示数据与给定模型的相似程度，该值越大，表示模型品质越高
		/*
		 * Saving model
		 */

		// 将模型应用到新数据，要随模型一起保存/加载管道，使用该管道添加新实例
		String modelPath = "myTopicModel";

		// 序列化保存对象
		// 保存模型
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(modelPath + ".model")));
		oos.writeObject(model);
		oos.close();
		// 保存流水线
		oos = new ObjectOutputStream(new FileOutputStream(new File(modelPath + ".pipeline")));
		oos.writeObject(pipeline);
		oos.close();

		System.out.println("Model saved.");

		/*
		 * Loading the model
		 */
		// 用ObjectInputStream类恢复模型
		// 加载模型
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(modelPath + ".model")));
		model = (ParallelTopicModel) ois.readObject();
		ois.close();
		// 加载流水线
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

		// Wallach提出的一种衡量模型质量的方法，该方法计算的是模型下抽取文档的对数概率，将未见过的文档的可能性用来比较模型-->可能性越高，模型越好
		System.out.println("Evaluation");

		// 划分数据集
		InstanceList[] instanceSplit = instances.split(new Randoms(), new double[] { 0.9, 0.1, 0.0 });

		// 使用前90%数据训练
		model.addInstances(instanceSplit[0]);
		model.setNumThreads(4);
		model.setNumIterations(50);
		model.estimate();

		// 获取评价器，实现了留存文档的Wallach对数概率
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
