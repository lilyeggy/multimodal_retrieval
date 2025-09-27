# FlickrParser.py
# This script aims to align the image with its entity bound box and captions
# it will generate an ID for each image-caption pair for the search engine Milvus

import os 
import json
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from collections import defaultdict


class FlickrParser:
    def __init__(self,image_dir,bbox_dir,caption_dir):
        self.image_dir = Path(image_dir)
        self.bbox_dir = Path(bbox_dir)
        self.caption_dir = Path(caption_dir)
        self.entity_pattern = re.compile(r'\[/EN#(?P<entity_id>\d+)/(?P<category>\w+)\s(?P<text_phrase>.*?)\]')

    def _parse_caption_line(self,line):
        """
        Args:
        The line in the caption file

        Returns:
        The full caption and the entities in the line

        We only parse a single line in the caption file here
        """
        entities = []
        # find all the matches and their position using finditer
        matches = list(self.entity_pattern.finditer(line))

        for match in matches:
            match_dict = match.groupdict(match)
            entities.append({
                "entity_id": match_dict['entity_id'],
                "category": match_dict['category'],
                "text_phrase": match_dict['text_phrase']
            })

        # get the caption without the annotations
        # re.sub will only replace the part it matches 
        # the part it doesn't match will stay the same
        full_caption = self.entity_pattern.sub(lambda m:m.group('text_phrase'), line)
        # re.sub(replacement,string) will replace the string with replacement
        full_caption = ' '.join(full_caption.split())

        return full_caption,entities
    
    def _parse_caption(self,caption_file):
        """
        Args:
        The caption file to be parsed 

        Returns:
        All the captions and information about the entities in the caption file

        """
        full_captions = []
        all_entities_by_caption = []
        with open(caption_file,'r',encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                full_caption,entities = self._parse_caption_line(line)
                full_captions.append(full_caption)
                all_entities_by_caption.append(entities)
        return full_captions,all_entities_by_caption
    
    def _process_bounding_box(self,bbox_file):
        """
        Args:
        The bounding box file to be processed

        Returns:
        The bounding box dict and the format is like this:
        {
            "entity_id": [xmin,xmax,ymin,ymax]
        }
        
        """
        tree = ET.parse(bbox_file)
        root = tree.getroot()
        bboxes = {}
        for obj_elem in root.findall('object'):
            entity_id = obj_elem.find('name').text
            bndbox_elem = obj_elem.find('bndbox')
            if bndbox_elem is not None:
                box = [
                    int(bndbox_elem.find('xmin').text),
                    int(bndbox_elem.find('ymin').text),
                    int(bndbox_elem.find('xmax').text),
                    int(bndbox_elem.find('ymax').text)
                ]
                bboxes[entity_id] = box
        return bboxes
    
    def process_single_image(self, image_id):
        """
        Args:
        The file name without file format

        Returns:
        Metadata of our data

        The format of metadata is :
        {
            "image_id" : [
                {
                    "type": "entity",
                    "image_id": image_name,
                    "entity_id": entity_id,
                    "category": entity['category'],
                    "text_phrase": entity['text_phrase'], 
                    "bounding_box": bboxes_by_id[entity_id] 
                },
                {
                    "type": "caption",
                    "image_id": image_name,
                    "caption_id": i,
                    "caption": caption
                }
            ]
        }
        """
        caption_path = self.caption_dir / f"{image_id}.txt"
        bbox_path = self.bbox_dir / f"{image_id}.xml"

        if not caption_path.exists() or not bbox_path.exists():
            print(f"Warning: Skipping {image_id} due to missing annotation files.")
            return []

        full_captions, all_entities_by_caption = self._parse_caption(caption_path)
        bboxes_by_id = self._process_bounding_box(bbox_path)
        
        metadata = {}
        image_name = f"{image_id}.jpg"

        metadata[image_name] = []
        
        for entities_in_one_caption in all_entities_by_caption:
            
            for entity in entities_in_one_caption:
                entity_id = entity['entity_id']
                image_id = image_name
                
                if entity_id in bboxes_by_id:
                    metadata[image_name].append({
                        "type": "entity",
                        "image_id": image_name,
                        "entity_id": entity_id,
                        "category": entity['category'],
                        "text_phrase": entity['text_phrase'], 
                        "bounding_box": bboxes_by_id[entity_id] 
                    })


        # add the full caption in the caption file
        for i, caption in enumerate(full_captions):
            metadata[image_name].append({
                "type": "caption",
                "image_id": image_name,
                "caption_id": i, # the i-th caption in the caption file
                "caption": caption
            })
            
        return metadata
    
    def parse_all_files(self):
        """
        Process all files and return the metadata
        """
        all_metadata = []
        image_files = sorted(self.image_dir.glob("*.jpg")) # 排序保证一致性
        for image_path in image_files:
            image_id = image_path.stem # 获取不带后缀的文件名
            all_metadata.extend(self.process_single_image(image_id))
        return all_metadata
    
    

