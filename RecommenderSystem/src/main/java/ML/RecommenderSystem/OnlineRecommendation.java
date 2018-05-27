/**
 * 
 */
package ML.RecommenderSystem;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.PlusAnonymousConcurrentUserDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * @XinCheng 2018年5月20日 Administrator 在线学习引擎，在线--》对于新注册用户的推荐，Mahout允许向数据模型添加临时用户
 *           1.使用当前数据定期重建整个推荐(每天或每小时，具体取决于耗费多长时间) 2.推荐时，检查系统中是否有这个用户
 *           3.若有，像往常一样结束推荐 4.若没有，则创建临时用户，填入偏好，做推荐
 */
public class OnlineRecommendation {

	Recommender recommender;
	int concurrentUsers = 100;
	int noItems = 10;

	public OnlineRecommendation() throws IOException {
		DataModel model = new StringItemIdFileDataModel(new File("data/BX-Book-Ratings.csv"), ";");
		// 对于临时用户，使用PlusAnonymousConcurrentUserDataModel类实例包装数据模型，该类允许获得一个临时用户ID，以后必须释放该ID，以便重用，得到ID，填写偏好，然后开始推荐
		PlusAnonymousConcurrentUserDataModel plusModel = new PlusAnonymousConcurrentUserDataModel(model,
				concurrentUsers);
		// recommender = ...;

	}

	public List<RecommendedItem> recommend(long userId, PreferenceArray preferences) throws TasteException {
		if (userExistsInDataModel(userId)) {
			return recommender.recommend(userId, noItems);
		} else {
			PlusAnonymousConcurrentUserDataModel plusModel = (PlusAnonymousConcurrentUserDataModel) recommender
					.getDataModel();
			// 轮询获取匿名用户ID
			Long anonymousUserID = plusModel.takeAvailableUser();
			// 设置临时偏好
			PreferenceArray tempPrefs = preferences;
			tempPrefs.setUserID(0, anonymousUserID);
			// tempPrefs.setItemID(0, itemID);
			plusModel.setTempPrefs(tempPrefs, anonymousUserID);

			List<RecommendedItem> results = recommender.recommend(anonymousUserID, noItems);
			// 轮询释放该用户ID
			plusModel.releaseUser(anonymousUserID);
			return results;
		}
	}

	private boolean userExistsInDataModel(long userId) {
		return false;
	}

}
