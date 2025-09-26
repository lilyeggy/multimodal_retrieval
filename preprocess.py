# preprocess.py
# This script aims to align the image with its entity bound box and captions
# it will generate an ID for each image-caption pair for the search engine Milvus

import os 
import json
import xml.etree.ElementTree as ET
import re





class Parser():
    def __init__(self,IMAGE_DIR="data/images/flickr30k-images",
                 BOUNDINGBOX_DIR="data/annotations/Annotations",
                 CAPTION_DIR="data/annotations/Sentences",
                 OUTPUT_DIR="data/preprocessed"):
        self.IMAGE_DIR = IMAGE_DIR
        self.BOUNDINGBOX_DIR = BOUNDINGBOX_DIR
        self.CAPTION_DIR = CAPTION_DIR
        self.OUTPUT_DIR = OUTPUT_DIR


    def extract_full_description(self,caption):
        """
        Args:
        The caption with annotations

        Returns:
        cleaned caption
        """
        # 这个正则表达式模式会匹配并“捕获”你想要保留的部分。
        # pattern = r'\[/EN#.*?/(.*?)\]'
        # 这个模式会匹配所有 [ 和 ] 之间的内容，并捕获斜杠后和右括号前的文字。

        # 使用 re.sub() 的函数参数进行替换
        def replacer(match):
            # match.group(1) 返回捕获组 (.*?) 中的内容，也就是名词短语
            elements = match.group(1).split(' ')
            matcher = " ".join(elements[1:])
            return matcher

        # 使用 re.sub() 配合上面定义的 replacer 函数进行替换
        cleaned_text = re.sub(r'\[/EN#.*?/(.*?)\]', replacer, caption)

        # 移除替换后可能产生的多余空格，并将多个空格替换为单个空格
        final_description = ' '.join(cleaned_text.split())

        return final_description
        
    def extract_entity_category(self,caption):
        """
        Args:
        The caption with annotations

        Returns:
        A list whose elements are the entities in the caption
        
        Return Examples:
        [[all the entities in caption 1],[all the entities in caption 2],...]
        """
        pattern = r'\[/EN#.*?/(.*?)\]'
        matches = re.findall(pattern,caption)
        entity_caption = []
        for i in matches:
            elements = i.split(' ')
            entity = elements[0]
            entity_caption.append(entity)
        return entity_caption
    
    def extract_entity_description(self,caption):
        """
        Args:
        The caption with annotations

        Returns:
        A list whose elements are the entities in the caption
        
        Return Examples:
        [[all the entities in caption 1],[all the entities in caption 2],...]
        """
        pattern = r'\[/EN#.*?/(.*?)\]'
        matches = re.findall(pattern,caption)
        entity_caption = []
        for i in matches:
            elements = i.split(' ')
            entity = " ".join(elements[1:])
            entity_caption.append(entity)
        return entity_caption

    def process_caption(self,caption_file):
        """
        Args: 
        The file path of the caption file

        Returns:
        The list of all captions
        """
        self.full_captions = []
        self.all_entity_caption = []
        self.all_entity_category = []
        with open(caption_file,'r',encoding='utf-8') as f:
            for i,line in enumerate(f):
                cleaned_caption = self.extract_full_description(line) # 获取完整的caption
                self.full_captions.append(cleaned_caption)
                entity_caption = self.extract_entity_description(line)
                self.all_entity_caption.append(entity_caption)
                entity_category = self.extract_entity_category(line)
                self.all_entity_category.append(entity_category)

        return self.full_captions,self.all_entity_caption,self.all_entity_category



    def process_bounding_box(self,bounding_box_file):
        """
        Args:
        The file path of the bounding box file

        Returns:
        The list of all bounding boxes

        Return Format:
        {
            "filename":   ,
            "image_size": {
                "width":,
                "height":
            },
            "objects":[
                {
                    "caption_id":  ,
                    "bounding_box": [xmin,xmax,ymin,ymax],
                    "has_bndbox":
                },
                ...,
                {
                    "caption_id":  ,
                    "bounding_box": [xmin,xmax,ymin,ymax],
                    "has_bndbox":
                }
            ]
        }
        """
        tree = ET.parse(bounding_box_file)
        root = tree.getroot()
        self.bounding_box = {}

        filename_elem = root.find('filename')
        if filename_elem is not None:
            self.bounding_box['filename'] = filename_elem

        size_elem = root.find('size')
        if size_elem is not None:
            self.bounding_box['image_size'] = {
                'width':int(size_elem.find('width').text),
                'height':int(size_elem.find('height').text)
            }
        
        self.bounding_box['objects'] = []

        for obj_elem in root.findall('object'):
            obj_info = {}
            
            name_elem = obj_elem.find('name')
            if name_elem is not None:
                obj_info['caption_id'] = name_elem.text

            bndbox_elem = obj_elem.find('bndbox')
            if bndbox_elem is not None:
                bndbox = [int(bndbox_elem.find('xmin').text),int(bndbox_elem.find('xmax').text),
                          int(bndbox_elem.find('ymin').text),int(bndbox_elem.find('ymax').text)]        
                obj_info['bounding_box'] = bndbox
                obj_info['has_bndbox'] = True
            
            else:
                obj_info['has_bndbox'] = False
            self.bounding_box['objects'].append(obj_info)
        
        return self.bounding_box

    def combine_bounding_box_and_caption(self):
        """
        Return Format:
        {
            "image_id": "1000092795.jpg",
            "caption_id": "137644",
            "category": "people",
            "text_phrase": "A group of friends",
            "bounding_box": [28, 147, 50, 138]
        },
        {
            "image_id":
            "caption":
        }
        """
        self.annotations_one_file = []
        self.num_entity = len(self.bounding_box['objects']) / len(self.all_entity_caption)
        for i in range(self.num_entity):
            for k in range(len(self.all_entity_caption)):
                # i * len(self.all_entity_caption) = 第 i 个实体
                annotations = {}
                annotations['image_id'] = self.bounding_box['filename']
                annotations['caption_id'] = self.bounding_box['objects'][i]['caption_id']
                annotations['bounding_box'] = self.bounding_box['objects'][i]['bounding_box']
                annotations['category'] = self.all_entity_category[k][i]
                annotations['text_phrase'] = self.all_entity_caption[k][i]
                self.annotations_one_file.append(annotations)

        for i in range(len(self.all_entity_caption)):
            annotations = {}
            annotations['image_id'] = self.bounding_box['filename']
            annotations['caption'] = self.full_captions[i]
            self.annotations_one_file.append(annotations)
        return self.annotations_one_file

    def parse_all_files(self):
        """
        Returns:
        The metadata after processing all the files
        """

        self.metadata = []
        for image_name in os.listdir(self.IMAGE_DIR):
            if image_name.endswith(".jpg"):
                image_id = image_name.split(".jpg")[0]

                caption_file_path = os.path.join(self.CAPTION_DIR, image_id + ".txt")
                if not os.path.exists(caption_file_path):
                    raise FileNotFoundError(f"There is no such file:{caption_file_path}")
                self.full_captions,self.all_entity_caption,self.all_entity_category = self.process_caption(caption_file_path)

                bouding_box_file_path = os.path.join(self.BOUNDINGBOX_DIR, image_id + ".xml")
                if not os.path.exists(bouding_box_file_path):
                    raise FileNotFoundError(f"There is no such file:{bouding_box_file_path}")
                bounding_box_data = self.process_bounding_box(bouding_box_file_path)    
                annotations = self.combine_bounding_box_and_caption()
                self.metadata.extend(annotations)
        return self.metadata




        



# def parse_annotations(annotation_file):
    