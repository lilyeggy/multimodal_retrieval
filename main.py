from extractor import Extractor
from milvus_manager import MilvusManager
from flickr_searcher import FlickrSearcher
from PIL import Image
import torch
def main():
    IMAGE_DIR = "../data/images/flickr30k-images"
    BBOX_DIR = "../data/bounding_boxes/flickr30k_bounding_boxes"
    CAPTION_DIR = "../data/captions/flickr30k_captions"

    MODEL_NAME = "Qwen/Qwen-VL"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dim = 768

    # =====  步骤 1 ：提取特征 ===========

    # 提取图像特征、实体特征和文本特征
    extractor = Extractor(IMAGE_DIR, BBOX_DIR, CAPTION_DIR, MODEL_NAME, device,embedding_dim)

    print("开始提取特征....")
    features = extractor.extract_all_features()
    print(f"特征提取完成，共提取到{len(features)}条数据")

    # 关闭提取器
    extractor.close()

    # ====   步骤 2 ：存入Milvus
    HOST = "127.0.0.1"
    PORT = "19530"
    COLLECTION_NAME = "flickr_multimodal"
    milvus_manager = MilvusManager(HOST,PORT,COLLECTION_NAME)
    # 启动Milvus连接
    milvus_manager.connect()
    # 创建Milvus集合
    # milvus_manager.create_collection() # 不需要，初始化的时候已经创建了
    # 插入数据
    milvus_manager.insert_embeddings(features)
    # 创建索引
    field_name = "image_embedding" # 索引字段名称
    milvus_manager.create_index(field_name=field_name)

    # 加载数据到内存
    milvus_manager.load_collection() 

    # 关闭连接
    # milvus_manager.close()

    # ==== 步骤 3 ：执行搜索 =========
    
    searcher = FlickrSearcher(COLLECTION_NAME,HOST,PORT,MODEL_NAME,device,embedding_dim)
    query_image = Image.open("../data/images/query_image.jpg").convert("RGB")
    query_text = "a boy is playing a ball game"

    # 1：图像到图像检索
    print("图像到图像检索")
    results = searcher.search_by_image(query_image,search_type="image_embedding")

    # 2：图像到文本检索
    print("图像到文本检索")
    results = searcher.search_by_image(query_image,search_type="text_embedding")

    # 3：文本到图像检索
    print("文本到图像检索")
    results = searcher.search_by_text(query_text,search_type="image_embedding")

    # 4：文本到文本检索
    print("文本到文本检索")
    results = searcher.search_by_text(query_text,search_type="text_embedding")

    # 关闭搜索器
    searcher.close()

if __name__ == "__main__":
    main()

