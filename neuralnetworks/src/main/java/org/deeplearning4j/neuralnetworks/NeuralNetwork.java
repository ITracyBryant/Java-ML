package org.deeplearning4j.neuralnetworks;

import java.util.Collections;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Hello world!
 *
 */
public class NeuralNetwork {
	private static Logger log = LoggerFactory.getLogger(NeuralNetwork.class);

	public static void main(String[] args) throws Exception {
		final int numRows = 28;
		final int numColumns = 28; // 28 * 28个像素
		int outputNum = 10; // 10个目标类
		int numSamples = 60000; // 60000个样本
		int batchSize = 100; // 迭代批大小
		int iterations = 10; // 迭代次数
		int seed = 123; // 随机种子数
		int listenerFreq = batchSize / 5; // 监听频率

		log.info("Load Data: ");
		// MNIST数据集，由手写数字组成，包含60000个训练和10000个测试图像
		// MNIST数据集加载器DataSetIterator类支持的数据集是impl包的一部分，有数据集Iris,MNIST等
		DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples, true); // 下载数据集和其标签，true参数表示将数据集二值化

		log.info("Build model: ");
		// MultiLayerNetwork model = softMaxRegression(seed, iterations,
		// numRows, numColumns, outputNum);
		// MultiLayerNetwork model = deepBeliefNetwork(seed, iterations,
		// numRows, numColumns, outputNum);
		MultiLayerNetwork model = deepConvNetwork(seed, iterations, numRows, numColumns, outputNum);

		model.init(); // 初始化生成模型
		model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq))); // 绑定模型与训练数据

		log.info("Train model: ");
		model.fit(iter); // 触发端对端的网络训练

		log.info("Evaluate model: ");
		@SuppressWarnings("rawtypes")
		Evaluation eval = new Evaluation(outputNum); // 评估模型，存储批结果

		// 在数据集上分批做迭代，以便让内存消耗保持在一个合理范围内，结果保存在一个eval对象中
		DataSetIterator testIter = new MnistDataSetIterator(100, 10000);
		while (testIter.hasNext()) {
			DataSet testMnist = testIter.next();
			INDArray predict = model.output(testMnist.getFeatureMatrix());
			eval.eval(testMnist.getLabels(), predict);
		}
		log.info(eval.stats());
		log.info("**************Example finished*****************");
	}

	// 创建单层回归模型，基于softmax激活函数，输入28*28=784个神经元。输出10个神经元目标，网络中层是全连接
	@SuppressWarnings("unused")
	private static MultiLayerNetwork softMaxRegression(int seed, int iterations, int numRows, int numColumns,
			int outputNum) {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder() // 定义神经网络
				.seed(seed) // 为梯度搜索定义参数
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0).iterations(iterations).momentum(0.5) // momentum参数指定优化算法收敛到局部最优的速度
				.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // 使用共轭梯度最优化算法迭代
				.list(1)
				.layer(0,
						new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // 指定网络有一个单层，定义损失函数(NEGATIVELOGLIKELIHOOD)
								.activation("softmax") // 内部感知器激活函数(softmax)
								.nIn(numRows * numColumns).nOut(outputNum).build())// 输入输出层数量，即总图像样本数和目标变量数
				.pretrain(true).backprop(false) // 为网络开启预训练，关闭反向传播
				.build(); // 创建未经训练的网络结构

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		return model;
	}

	// 创建一个基于受限玻尔兹曼机的深度信念网络，由4层组成，第一层将0层784个神经元缩小为500个神经元，继续下层缩小道250，继续缩小到200，最后输出10个目标值。训练DBN时间更长，准确度也更高
	@SuppressWarnings("unused")
	private static MultiLayerNetwork deepBeliefNetwork(int seed, int iterations, int numRows, int numColumns,
			int outputNum) {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0).iterations(iterations).momentum(0.5)
				.momentumAfter(Collections.singletonMap(3, 0.9))
				.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // 定义梯度优化算法
				.list(4) // 指定网络有4个层
				.layer(0,
						new RBM.Builder().nIn(numRows * numColumns).nOut(500).weightInit(WeightInit.XAVIER) // 使用均方根误差交叉熵Xavier算法初始化权重
								.lossFunction(LossFunction.RMSE_XENT).visibleUnit(RBM.VisibleUnit.BINARY)
								.hiddenUnit(RBM.HiddenUnit.BINARY).build()) // 基于输入和输出神经元数目自动确定初始化权重范围
				.layer(1,
						new RBM.Builder().nIn(500).nOut(250).weightInit(WeightInit.XAVIER)
								.lossFunction(LossFunction.RMSE_XENT).visibleUnit(RBM.VisibleUnit.BINARY)
								.hiddenUnit(RBM.HiddenUnit.BINARY).build()) // 该层拥有相同参数，但输入输出神经元数目不同
				.layer(2,
						new RBM.Builder().nIn(250).nOut(200).weightInit(WeightInit.XAVIER)
								.lossFunction(LossFunction.RMSE_XENT).visibleUnit(RBM.VisibleUnit.BINARY)
								.hiddenUnit(RBM.HiddenUnit.BINARY).build()) // 该层拥有相同参数，但输入输出神经元数目不同
				.layer(3,
						new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax").nIn(200)
								.nOut(outputNum).build()) // 最后一层将神经元映射到输出，使用softmax激活函数
				.pretrain(true).backprop(false).build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		return model;
	}

	// 创建多层卷积神经网络，由7层组成，先用max
	// pooling重复两对卷积和子采样层，将最后一个子采样层连接到前馈神经网络，最后三层依次含有120，84，10个神经元。(完整的图像识别管道，前4个图层用于特征提取，后3个图层用于学习模型)
	// 训练时间更长，但准确度更高再98%左右，模型主要依赖于线性代数运算，使用GPU可以明显提高训练速度
	private static MultiLayerNetwork deepConvNetwork(int seed, int iterations, int numRows, int numColumns,
			int outputNum) {
		MultiLayerConfiguration.Builder conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
				.activation("sigmoid").weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.0, 0.01)) // 权重服从正态分布。激活函数sigmoid
				// .learningRate(7*10e-5)
				.learningRate(1e-3).learningRateScoreBasedDecayRate(1e-1)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // 指定随机梯度下降算法及其参数
				.list(7) // 指定7个网络层
				.layer(0,
						new ConvolutionLayer.Builder(new int[] { 5, 5 }, new int[] { 1, 1 }).name("cnn1")
								.nIn(numRows * numColumns).nOut(6).build()) // 第一个卷积层输入一幅完整图像，输出6个特征图，卷积层应用5*5过滤器(卷积)，结果存储在1*1单元格中
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 },
								new int[] { 2, 2 }).name("maxpool1").build()) // 第二层子采样层，接收2*2区域，把最大结果存为2*2
				// 接下来的两层重复前两层
				.layer(2,
						new ConvolutionLayer.Builder(new int[] { 5, 5 }, new int[] { 1, 1 }).name("cnn2").nOut(16)
								.biasInit(1).build())
				.layer(3,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] { 2, 2 },
								new int[] { 2, 2 }).name("maxpool2").build())
				.layer(4, new DenseLayer.Builder().name("ffn1").nOut(120).build()) // 然后将子采样层的输出连接到稠密前馈网络，先是输出120个神经元
				.layer(5, new DenseLayer.Builder().name("ffn2").nOut(84).build()) // 再穿过一层变为输出84个神经元
				.layer(6,
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
								.nOut(outputNum).activation("softmax") // 需要径向基函数(radial
																		// basis
																		// function)
								.build())
				.backprop(true).pretrain(false).cnnInputSize(numRows, numColumns, 1); // 最后一层将84个神经元与10个输出神经元连接在一起

		MultiLayerNetwork model = new MultiLayerNetwork(conf.build());

		return model;
	}
}
