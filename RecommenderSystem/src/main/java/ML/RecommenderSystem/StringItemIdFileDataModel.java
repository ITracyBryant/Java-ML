/**
 * 
 */
package ML.RecommenderSystem;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;

/**
 * @XinCheng 2018年5月20日 Administrator
 *
 */
public class StringItemIdFileDataModel extends FileDataModel {

	// 初始化将String转换为long的转换器
	public ItemMemIDMigrator memIdMigtr;

	public StringItemIdFileDataModel(File dataFile, String delimiterRegex) throws IOException {
		super(dataFile, delimiterRegex);
		// TODO Auto-generated constructor stub
	}

	@Override
	protected long readItemIDFromString(String value) {
		if (memIdMigtr == null) {
			memIdMigtr = new ItemMemIDMigrator();
		}
		// 转换为long
		long retValue = memIdMigtr.toLongID(value);
		// 存储到缓存中
		if (null == memIdMigtr.toStringID(retValue)) {
			try {
				memIdMigtr.singleInit(value);
			} catch (TasteException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return retValue;
	}

	// 将long转换为String
	String getItemIDAsString(long itemId) throws TasteException {
		return memIdMigtr.toStringID(itemId);
	}

}
