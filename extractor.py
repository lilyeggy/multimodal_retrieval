from FlickrParser import FlickrParser

from pathlib import Path
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, AutoTokenizer,AutoModelForCausalLM
from typing import List,Dict,Any

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
        """
        if embedding.shape[0] == self.embedding_dim:
            return embedding
        elif embedding.shape[0] < self.embedding_dim:
            padded = torch.zeros(self.embedding_dim)
            padded[:embedding.shape[0]] = embedding
            return padded
        else:
            return embedding[:self.embedding_dim]
