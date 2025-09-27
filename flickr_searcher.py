from pymilvus import connections,Collection
from typing import List,Dict,Any
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM,AutoTokenizer

class FlickrSearcher:
    def __init__(self,collection_name,host="127.0.0.1",port=19530,model_name="Qwen/Qwen-VL",device=None,
                 embedding_dim=768):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim

        print(f"连接Milvus服务到{host}:{port}")
        connections.connect(host=host,port=port)

        # 获取集合
        self.collection = Collection(self.collection_name)

        # 初始化模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 加载集合到内存
        self.collection.load()
        print("已经加载集合到内存")

    def _adjust_embedding_dim(self, embedding: torch.Tensor) -> torch.Tensor:
        """调整向量维度到目标维度"""
        if embedding.shape[0] == self.embedding_dim:
            return embedding
        elif embedding.shape[0] < self.embedding_dim:
            padded = torch.zeros(self.embedding_dim)
            padded[:embedding.shape[0]] = embedding
            return padded
        else:
            return embedding[:self.embedding_dim]
            
    
    def search_by_image(self,
                        query_image:Image.Image,
                        search_type="image_embedding",
                        top_k=5,
                        filter_expr=None
                        )->List[Dict[str,Any]]:
        """
        基于图像搜索
        Args:
        query_image: 查询的图像
        search_type: 搜索的类型，可以是"image_embedding"或者"text_embedding"
        top_k: 返回的搜索结果数量
        filter_expr: 过滤条件比如"type=='entity'"

        """
        image_inputs = self.processor(query_image,return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_outputs = self.model.forward(**image_inputs,output_hidden_states=True)
            query_embedding = image_outputs.hidden_states[-1]
            query_embedding = torch.mean(query_embedding,dim=1).squeeze(0)
            query_embedding = self._adjust_embedding_dim(query_embedding)

        query_embedding_list = query_embedding.cpu().numpy().tolist()

        # 搜索参数
        search_params = {
            "metric_type": "IP", # 搜索方式
            # 构建索引的时候使用的metric_type要和搜索的时候的metric_type一致
            "params": {"nprobe": 10},  
            # nprobe: 搜索的节点数量，设置小可以认为是粗略地搜索，追求速度，设置的大就是要求精细搜索
        }

        # 执行搜索
        results = self.collection.search(
            data=[query_embedding_list],
            anns_field=search_type,
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=[
                "type",
                "image_id",
                "text_phrase",
                "caption",
                "category",
                "bounding_box"
            ]
        )

        parsed_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "type": hit.entity.get('type'),
                    "image_id": hit.entity.get('image_id'),
                    "text_phrase": hit.entity.get('text_phrase'),
                    "caption": hit.entity.get('caption'),
                    "category": hit.entity.get('category'),
                    "bounding_box": hit.entity.get('bounding_box')
                }
                parsed_results.append(result)
        
        print(f"基于图像的搜索结果（搜索字段：{search_type}）")

        for i, res in enumerate(parsed_results[:top_k]):
            print(f"  {i+1}. ID: {res['id']}, 距离: {res['distance']:.4f}")
            print(f"     类型: {res['type']}, 图像: {res['image_id']}")
            if res['text_phrase']:
                print(f"     文本短语: {res['text_phrase']}")
            if res['caption']:
                print(f"     完整描述: {res['caption']}")
            print("  ---")

        return parsed_results

    def search_by_text(self,query_text:str,
                       search_type="text_embedding",
                       top_k=5,
                       filter_expr=None
                       ):
        """
        基于文本搜索
        Args:
        query_text: 查询的文本
        search_type: 搜索的类型，可以是"image_embedding"或者"text_embedding"
        top_k: 搜索结果数量
        filter_expr: 过滤条件比如"type=='entity'"

        """
        text_inputs = self.tokenizer(text=query_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.forward(**text_inputs, output_hidden_states=True)
            query_embedding = outputs.hidden_states[-1]
            query_embedding = torch.mean(query_embedding, dim=1).squeeze(0)
            query_embedding = self._adjust_embedding_dim(query_embedding)
        query_embedding_list = query_embedding.cpu().numpy().tolist()

        search_params = {
            "metric_type": "IP", # 搜索方式
            "params": {"nprobe": 10},  # 搜索参数
        }

        # 执行搜索
        results = self.collection.search(
            data=[query_embedding_list],
            anns_field=search_type,
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=[
                "type",
                "image_id",
                "text_phrase",
                "caption",
                "category",
                "bounding_box"
            ]
        )       


        parsed_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "type": hit.entity.get('type'),
                    "image_id": hit.entity.get('image_id'),
                    "text_phrase": hit.entity.get('text_phrase'),
                    "caption": hit.entity.get('caption'),
                    "category": hit.entity.get('category'),
                    "bounding_box": hit.entity.get('bounding_box')
                }
                parsed_results.append(result)
        
        print(f"基于图像的搜索结果（搜索字段：{search_type}）")

        for i, res in enumerate(parsed_results[:top_k]):
            print(f"  {i+1}. ID: {res['id']}, 距离: {res['distance']:.4f}")
            print(f"     类型: {res['type']}, 图像: {res['image_id']}")
            if res['text_phrase']:
                print(f"     文本短语: {res['text_phrase']}")
            if res['caption']:
                print(f"     完整描述: {res['caption']}")
            print("  ---")

        return parsed_results
    
    def close(self):
        """关闭连接"""
        connections.disconnect("default")
        # 清理模型
        del self.model
        del self.tokenizer
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
