from FlickrParser import FlickrParser

from pathlib import Path
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, AutoTokenizer,AutoModelForCausalLM
from typing import List,Dict,Any,Optional,Tuple

class Extractor:
    def __init__(self,
                 image_dir: str,
                 bbox_dir: str,
                 caption_dir: str,
                 model_name: str,
                 device: str = None,
                 embedding_dim: int = 768):
        
        self.image_dir = image_dir
        self.bbox_dir = bbox_dir
        self.caption_dir = caption_dir
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim

        # 初始化解析器
        self.parser = FlickrParser(image_dir, bbox_dir, caption_dir)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    def _adjust_embedding_dim(self,embedding:torch.Tensor)->torch.Tensor:
        """
        调整向量的维度到目标维度
        Args:
            embedding (torch.Tensor): 嵌入向量

        Returns:
            torch.Tensor: 调整后的嵌入向量
        """
        if embedding.shape[0] == self.embedding_dim:
            return embedding
        elif embedding.shape[0] < self.embedding_dim:
            padded = torch.zeros(self.embedding_dim)
            padded[:embedding.shape[0]] = embedding
            return padded
        else:
            return embedding[:self.embedding_dim]
        
    def _crop_image(self,image:Image.Image,bbox:List[int]) -> Image.Image:
        """
        根据bounding box裁剪图像
        Args:
            image (Image.Image): 图像
            bbox (List[int]): bounding box

        Returns:
            Image.Image: 裁剪后的图像
        """
        left,top,right,bottom = bbox
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        width,height = image.size
        left = max(0,left)
        top = max(0,top)
        right = min(width,right)
        bottom = min(height,bottom)
        cropped_image = image.crop((left,top,right,bottom))
        return cropped_image

    def _extract_embeddings(self,image:Image.Image,text:str)->Tuple[torch.Tensor,torch.Tensor] :
        """
        从文本和图像中提取嵌入向量
        Args:
            image (Image.Image): 图像
            text (str): 文本

        Returns:
            Tuple[torch.Tensor,torch.Tensor]: 嵌入向量
        """
        image_inputs = self.processor(image,return_tensors="pt").to(self.device)
        text_inputs = self.tokenizer(text,return_tensors="pt").to(self.device)

        with torch.no_grad(): # 禁用梯度
            image_outputs = self.model.forward(**image_inputs,output_hidden_states=True)
            image_embedding = image_outputs.hidden_states[-1]
            image_embedding = torch.mean(image_embedding,dim=1).squeeze(0)
            image_embedding = self._adjust_embedding_dim(image_embedding)

            text_outputs = self.model.forward(**text_inputs,output_hidden_states=True)
            text_embedding = text_outputs.hidden_states[-1]
            text_embedding = torch.mean(text_embedding,dim=1).squeeze(0)
            text_embedding = self._adjust_embedding_dim(text_embedding)

        return image_embedding,text_embedding
    
    def extract_all_features(self) -> List[Dict[str,Any]]:
        """
        提取所有特征
        Returns:
            List[Dict[str,Any]]: 所有特征
        """
        # 获取所有元数据
        print("开始解析所有文件，获取metadata")
        metadata = self.parser.parse_all_files()

        all_features = []

        # 获取所有图像文件
        image_files = sorted(self.image_dir.glob("*.jpg"))
        print("开始提取所有图像文件的特征")

        for idx,image_file in enumerate(image_files):
            image = Image.open(image_file).convert("RGB")
            image_id = image_file.stem # 获取文件名

            # 获取图像对应的所有实体及实体对应的描述

            entities_in_image = []
            for item in metadata:
                # 判断item是否符合我们的格式要求以及当前image_id在metadata里是否存在
                if isinstance(item,dict) and image_id in item:
                    entities_in_image = item[image_id]
                    break
            
            for entity in entities_in_image:
                # 对于实体：
                if "bounding_box" in entity:
                    bounding_box = entity["bounding_box"]
                    cropped_image = self._crop_image(image,bounding_box)
                    image_embedding,text_embedding = self._extract_embeddings(
                        cropped_image,entity["text_phrase"])
                    feature = {
                        "type":"entity",
                        "image_id":image_id,
                        "entity_id":entity["entity_id"],
                        "entity_category":entity["category"],
                        "entity_description":entity["text_phrase"],
                        "bounding_box":bounding_box,
                        "image_embedding":image_embedding.cpu().numpy(),
                        "text_embedding":text_embedding.cpu().numpy()
                    }
                    all_features.append(feature)
                
                # 处理完整的caption
                else:
                    caption = entity["caption"]
                    image_embedding,text_embedding = self._extract_embeddings(image,caption)
                    feature = {
                        "type": "caption",
                        "image_id": entity["image_id"],
                        "caption_id": entity["caption_id"],
                        "caption": entity["caption"],
                        "image_embedding":image_embedding.cpu().numpy(),
                        "text_embedding":text_embedding.cpu().numpy()
                    }
                    all_features.append(feature)

        print(f"特征提取完成，共提取{len(all_features)}个特征")
        return all_features
    
    def close(self):
        """
        关闭模型，清理模型占用的内存
        """
        del self.model
        del self.tokenizer
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

                
                


    
