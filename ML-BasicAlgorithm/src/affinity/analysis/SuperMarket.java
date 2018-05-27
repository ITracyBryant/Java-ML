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
 * @XinCheng 2018年5月18日 Administrator 购物篮分析，关联性规则学习，Apriori算法和FP(频繁模式)-增长算法
 *           "购物篮"--》服务与产品--> 医疗诊断，蛋白质序列，人口普查数据，客户关系管理(CRM)，IT运营分析
 *           最关键的部分在于对结果的分析，解释规则(相关性)
 */
public class SuperMarket {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		// 加载数据
		Instances data = new Instances(new BufferedReader(new FileReader("data/supermarket.arff")));
		// 构建模型，初始化Apriori实例 ，Apriori算法不断减小最小支持度，直到找到所需的规则数，带有给定的最小置信度
		Apriori model = new Apriori();
		model.buildAssociations(data); // 开始频繁模式挖掘
		System.out.println(model);
		System.out.println("****************************");
		// 构建FP-增长算法模型
		FPGrowth fpgModel = new FPGrowth();
		fpgModel.buildAssociations(data);
		System.out.println(fpgModel);
		/* 在处理大型数据集时，使用FP-增长算法需要的时间明显更短，执行效率更高 */
	}

}
