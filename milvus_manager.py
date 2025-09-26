from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
from typing import List, Dict, Any

class MilvusManager:
    def __init__(self, host="127.0.0.1", port="19530", collection_name="flickr_multimodal"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        # 1. 连接到 Milvus
        self.connect()
        # 2. 创建Milvus集合
        self.create_collection()

    def connect(self):
        """连接到 Milvus"""
        connections.connect(host=self.host, port=self.port)
        print(f"已连接到 Milvus 服务器 {self.host}:{self.port}")

    def close(self):
        """断开Milvus连接"""
        connections.disconnect("default")
        print("已断开 Milvus 连接")
    def create_collection(self):
        """创建 Milvus 集合"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"旧集合 {self.collection_name} 已删除")

        fields = [
            FieldSchema(name="id",dtype=DataType.INT64,is_primary=True,auto_id=True),
            FieldSchema(name="type",dtype=DataType.VARCHAR,max_length=64),
            FieldSchema(name="image_id",dtype=DataType.VARCHAR,max_length=256), 
            FieldSchema(name="entity_id",dtype=DataType.VARCHAR,max_length=64,nullable=True), # 可为空，完整的caption没有entity_id
            FieldSchema(name="text_phrase",dtype=DataType.VARCHAR,max_length=512,nullable=True),
            FieldSchema(name="caption",dtype=DataType.VARCHAR,max_length=1024,nullable=True),
            FieldSchema(name="category",dtype=DataType.VARCHAR,max_length=64,nullable=True),
            FieldSchema(name="bounding_box",dtype=DataType.JSON,nullable=True),
            FieldSchema(name="image_embedding",dtype=DataType.FLOAT_VECTOR,dim=768),
            FieldSchema(name="text_embedding",dtype=DataType.FLOAT_VECTOR,dim=768),
        ]
        schema = CollectionSchema(fields=fields, description="flickr 30k 多模态检索库")
        self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"已创建集合 {self.collection_name}")

    def insert_embeddings(self,data_list:List[Dict[str,Any]]):
        """插入数据到 Milvus 集合"""
        if not data_list: 
            print("无可插入数据")
            return

        # 按照字段顺序插入数据
        ids = []
        types = []
        image_ids = []
        entity_ids = []
        text_phrases = []
        captions = []
        categories = []
        bounding_boxes = []
        image_embeddings = []
        text_embeddings = []

        for data in data_list:
            ids.append(data["id"])
            types.append(data["type"])
            image_ids.append(data["image_id"])
            entity_ids.append(data["entity_id"])
            text_phrases.append(data["text_phrase"])
            captions.append(data["caption"])
            categories.append(data["category"])

            # Milvus要求向量一定要是list 或 numpy.array
            img_emb = data_list["image_embedding"]
            txt_embd = data_list["text_embedding"]

            # 检查img_emb和text_emb的类型
            if hasattr(img_emb,'cpu'):
                img_emb = img_emb.cpu().numpy()
            if hasattr(txt_embd,'cpu'):
                txt_embd = txt_embd.cpu().numpy()

            if hasattr(img_emb,'tolist'):
                img_emb = img_emb.tolist()
            if hasattr(txt_embd,'tolist'):
                txt_embd = txt_embd.tolist()

            image_embeddings.append(img_emb)
            text_embeddings.append(txt_embd)

        # 插入数据
        entities = [
            types,image_ids,entity_ids,text_phrases,captions,categories,bounding_boxes,
            image_embeddings,text_embeddings
        ]

        insert_result = self.collection.insert(entities)
        print(f"已插入 {len(insert_result.primary_keys)} 条数据")
        return insert_result.primary_keys
    
    def create_index(self,field_name="image_embedding"):
        """
        为向量字段创建索引
        向量字段包括：image_embedding, text_embedding
        """
        index_params = {
            "index_type": "IVF_FLAT", # 采用IVF索引
            "metric_type": "IP",  # 余弦相似度，适合归一化后的向量
            "params": {"nlist": 128} # 索引列表大小
        }
        self.collection.create_index(
            field_name=field_name,
            index_params=index_params
        )

    def load_collection(self):
        """加载集合数据"""
        self.collection.load()
        print(f"已加载集合，可以进行搜索")


    def search_embeddings(self,query_embedding,search_field="image_embedding", top_k=5, search_type="image"):
        """
        搜索相似向量

        Args:
            query_embedding (list or numpy.array): 查询向量
            search_field (str, optional): 搜索的字段.  "image_embedding" or "text_embedding".
            top_k (int, optional): 返回的相似向量的数量. Defaults to 5.
            search_type (str, optional): 搜索的类型. "image" or "text". Defaults to "image".
        """
        # 先转化为list或numpy.array，这样才能搜索
        if hasattr(query_embedding,'cpu'):
            query_embedding = query_embedding.cpu().numpy()
        if hasattr(query_embedding,'tolist'):
            query_embedding = query_embedding.tolist()

        search_params = {
            "metric_type": "IP", # 余弦相似度，适合归一化后的向量
            "params": {"nprobe": 10} # 搜索的索引列表大小
        }

        # 执行搜索
        results = self.collection.search(
            data=[query_embedding],
            anns_field=search_field,
            param=search_params,
            limit=top_k,
            output_fields=["type", "image_id", "text_phrase", "caption", "category", "bounding_box"]
        )

        print(f"搜索结果(基于{search_type}向量,搜索字段:{search_field})")

        for i,hits in enumerate(results):
            for hit in hits:
                print(f"  ID: {hit.id}, 距离: {hit.distance:.4f}")
                print(f"     类型:{hit.entity.get('type')}")
                print(f"     图片ID:{hit.entity.get('image_id')}")
                # 是实体就get实体的phrase，是图片就get图片的caption
                if hit.entity.get("text_phrase"):
                    print(f"     文本短语:{hit.entity.get('text_phrase')}")
                if hit.entity.get("caption"):
                    print(f"     描述:{hit.entity.get('caption')}")
                if hit.entity.get("category"):
                    print(f"     类别:{hit.entity.get('category')}")
                if hit.entity.get("bounding_box"):
                    print(f"     边界框:{hit.entity.get('bounding_box')}")
                print("-----------------------------------------")

        return results
    
    
                
                
