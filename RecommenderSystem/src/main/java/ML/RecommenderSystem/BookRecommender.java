package ML.RecommenderSystem;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericUserPreferenceArray;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.model.jdbc.MySQLJDBCDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import com.mysql.jdbc.jdbc2.optional.MysqlDataSource;

/**
 * 图书推荐引擎
 *
 */
public class BookRecommender implements RecommenderBuilder {
	static HashMap<String, String> books;

	public static void main(String[] args) throws Exception {
		books = loadBooks("data/BX-Books.csv");
		// itemBased();
		userBased();
		evaluateRecommender();

	}

	public static HashMap<String, String> loadBooks(String filename) throws Exception {
		HashMap<String, String> map = new HashMap<String, String>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		String line = "";
		while ((line = in.readLine()) != null) {
			String parts[] = line.replace("\"", "").split(";");
			map.put(parts[0], parts[1]);
		}
		in.close();
		// System.out.println(map.toString());
		System.out.println("Total Items: " + map.size());
		return map;
	}

	// 基于项目的过滤（基于项目的推荐引擎能够充分利用项目本身之间的关系），将计算建立在项目的相似性上(项目的相似性相对稳定，可以预先计算，无须实时重新计算)
	public static ItemBasedRecommender itemBased() throws Exception {
		// 加载数据
		StringItemIdFileDataModel dataModel = loadFromFile("data/BX-Book-Ratings.csv", ";");
		ItemSimilarity itemSimilarity = new PearsonCorrelationSimilarity(dataModel);
		ItemBasedRecommender recommender = new GenericItemBasedRecommender(dataModel, itemSimilarity);

		IDRescorer rescorer = new MyRescorer();
		// List recommendations = recommender.recommend(2, 3, rescorer);
		String itemISBN = "042513976X";
		long itemID = dataModel.readItemIDFromString(itemISBN);
		int noItems = 10;

		System.out.println("Recommendations for item: " + books.get(itemISBN));

		System.out.println("\nMost similar items: ");
		List<RecommendedItem> recommendations = recommender.mostSimilarItems(itemID, noItems);
		for (RecommendedItem recommendedItem : recommendations) {
			itemISBN = dataModel.getItemIDAsString(recommendedItem.getItemID());
			System.out.println("Item: " + books.get(itemISBN) + " | Item id: " + itemISBN + " | Value: "
					+ recommendedItem.getValue());
		}
		return recommender;

	}

	public static StringItemIdFileDataModel loadFromFile(String filePath, String seperator) throws Exception {
		StringItemIdFileDataModel dataModel = new StringItemIdFileDataModel(new File(filePath), seperator);
		return dataModel;
	}

	public static DataModel loadFromFile(String filePath) throws IOException {
		// 基于文件的DataModel，FileDataModel
		DataModel dataModel = new FileDataModel(new File("preferences.csv"));
		return dataModel;
	}

	public static DataModel loadFromDB() throws Exception {
		// 基于数据库的DataModel，MySQLJDBCDataModel
		/*
		 * A JDBCDataModel backed by a PostgreSQL database and accessed via
		 * JDBC. It may work with other JDBC databases. By default, this class
		 * assumes that there is a DataSource available under the JNDI name
		 * "jdbc/taste", which gives access to a database with a
		 * "taste_preferences" table with the following schema: CREATE TABLE
		 * taste_preferences ( user_id BIGINT NOT NULL, item_id BIGINT NOT NULL,
		 * preference REAL NOT NULL, PRIMARY KEY (user_id, item_id) ) CREATE
		 * INDEX taste_preferences_user_id_index ON taste_preferences (user_id);
		 * CREATE INDEX taste_preferences_item_id_index ON taste_preferences
		 * (item_id);
		 */
		MysqlDataSource dbsource = new MysqlDataSource();
		dbsource.setUser("user");
		dbsource.setPassword("pass");
		dbsource.setServerName("localhost");
		dbsource.setDatabaseName("my_db");

		DataModel dataModelDB = new MySQLJDBCDataModel(dbsource, "taste_preferences", "user_id", "item_id",
				"preference", "timestamp");
		return dataModelDB;
	}

	public DataModel loadInMemory() {
		// 基于内存的DataModel，GenericDataModels 数据模型也可以在内存中动态创建并保存
		FastByIDMap<PreferenceArray> preferences = new FastByIDMap<PreferenceArray>();
		// 为用户新建一个偏好数组，存储用户评分
		PreferenceArray prefsForUser = new GenericUserPreferenceArray(10); // 初始化偏好数组，必须给出占用内存大小的参数
		prefsForUser.setUserID(0, 1L);
		prefsForUser.setItemID(0, 101L);
		prefsForUser.setValue(0, 3.0f);
		prefsForUser.setItemID(1, 102L);
		prefsForUser.setValue(1, 4.5F);
		preferences.put(1L, prefsForUser); // 用userID作为key，添加用户偏好到散列映射
		// ... add others users（为多个用户添加多个偏好）
		// 返回偏好作为新的数据模型(使用偏好散列映射初始化GenericDataModel)
		DataModel dataModel = new GenericDataModel(preferences);
		return dataModel;
	}

	// 基于用户的协同过滤器(基于拥有相似行为的用户)
	public static void userBased() throws Exception {
		StringItemIdFileDataModel model = loadFromFile("data/BX-Book-Ratings.csv", ";");
		UserSimilarity similarity = new PearsonCorrelationSimilarity(model); // 定义计算用户关联性的方法，使用皮尔逊相关系数
		// 定义如何指出哪些用户是相似的，评分彼此相近的用户
		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
		// 使用数据模型，邻居，相似对象初始化GenericUserBasedRecommender默认引擎
		UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

		IDRescorer rescorer = new MyRescorer();

		// List recommendations = recommender.recommend(2, 3, rescorer);
		long userID = 276704; // 276704;//212124;//277157;
		int noItems = 10;

		System.out.println("Rated items by user: ");
		for (Preference preference : model.getPreferencesFromUser(userID)) {
			// convert long itemID back to ISBN
			String itemISBN = model.getItemIDAsString(preference.getItemID());
			System.out.println(
					"Item: " + books.get(itemISBN) + " | Item id: " + itemISBN + " | Value: " + preference.getValue());
		}

		System.out.println("\nRecommended items: ");
		List<RecommendedItem> recommendations = recommender.recommend(userID, noItems); // 返回一个IDRescorer实例
		for (RecommendedItem recommendedItem : recommendations) {
			String itemISBN = model.getItemIDAsString(recommendedItem.getItemID());
			System.out.println("Item: " + books.get(itemISBN) + " | Item id: " + itemISBN + " | Value: "
					+ recommendedItem.getValue());
		}
	}

	// 准确检测推荐有效程度的唯一方法，是在拥有实际用户的真实系统中进行A/B测试(A组收到一个随机推荐项目，B组收到推荐引擎推荐项目)
	// 实际操作过程中，可以使用脱机统计评估来估计，使用k折交叉验证，或使用Mahout实现的RecommenderEvaluator类，该类将数据集划分成两部分，第一部分默认为数据90%(用于推荐)，其余部分与评估值比较，测试匹配效果
	// RecommenderEvaluator类不直接接收recommender对象，需创建一个类实现RecommenderBuilder接口
	public static void evaluateRecommender() throws Exception {
		StringItemIdFileDataModel dataModel = loadFromFile("data/BX-Book-Ratings.csv", ";");
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder builder = new BookRecommender();
		double result = evaluator.evaluate(builder, null, dataModel, 0.9, 1.0); // 返回一个double值，0表示评估效果最佳，完美匹配用户偏好(值越小，匹配的越好)
		System.out.println(result);
	}

	// 创建一个实现RecommenderBuilder接口的类需要实现buildRecommender(),会返回一个recommender对象
	public Recommender buildRecommender(DataModel arg0) {
		try {
			return BookRecommender.itemBased();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
}
