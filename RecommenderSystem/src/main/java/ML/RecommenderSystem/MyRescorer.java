/**
 * 
 */
package ML.RecommenderSystem;

import org.apache.mahout.cf.taste.recommender.IDRescorer;

/**
 * @XinCheng 2018年5月20日 Administrator 添加自定义规则到推荐引擎
 */
public class MyRescorer implements IDRescorer {

	public boolean isFiltered(long arg0) {
		return false;
	}

	public double rescore(long itemId, double originalScore) {
		// double newScore = originalScore;
		if (bookIsNew(itemId)) {
			originalScore *= 1.3;
		}
		return Math.random();
	}

	public boolean bookIsNew(long itemId) {
		return false;
	}

}
