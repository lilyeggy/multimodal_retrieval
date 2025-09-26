# feature_extract.py
# this file aims to get the cropped image embedding and the full image embedding 
# for the milvus retrieval 
import os
from FlickrParser import FlickrParser
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoProcessor
from pathlib import Path
from PIL import Image
import torch

IMAGE_DIR = "data/images/flickr30k-images"
BOUNDINGBOX_DIR = "data/annotations/Annotations"
CAPTION_DIR = "data/annotations/Sentences"
OUTPUT_DIR = "data/preprocessed"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

parser = FlickrParser(IMAGE_DIR,BOUNDINGBOX_DIR,CAPTION_DIR)
metadata = parser.parse_all_files()

# 构建image_id -> entity 和 image_id -> caption 的索引

MODEL_NAME = "Qwen/Qwen-VL"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device,
    trust_remote_code=True
    ).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_NAME,trust_remote_code=True)


# get the image features of the whole image and the entities in images

image_files = sorted(IMAGE_DIR.glob("*.jpg"))

def crop_image(image,bounding_box):
    left,top,right,bottom = bounding_box
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    # avoid crossing the edge
    width,height = image.size
    left = max(0,left)
    top = max(0,top)
    right = min(width,right)
    bottom = min(height,bottom)

    cropped_image = image.crop((left,top,right,bottom))
    return cropped_image


all_vision_embeddings = []
all_entity_embeddings = []
for image_file in image_files:
    # get the image for cropping and embedding
    image = Image.open(image_file).convert("RGB")
    # get the captions 
    image_id = image_file.stem # get the file name without .jpg
    all_entity = metadata[image_id]
    for entity in all_entity:
        # for the entity
        if "bounding_box" in entity:
            bounding_box = entity["bounding_box"]
            bounding_image = crop_image(bounding_box)
            part_inputs = processor(image=bounding_image,return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.forward(**part_inputs,output_hidden_states=True)
                vision_embedding = outputs.hidden_states[0]
                entity["crop_image_embedding"] = vision_embedding
                # 这里可以直接构造entity
                # now entity changes into 
                """
                    {
                        "type": "entity",
                        "image_id": image_name,
                        "entity_id": entity_id,
                        "category": entity['category'],
                        "text_phrase": entity['text_phrase'], 
                        "bounding_box": bboxes_by_id[entity_id],
                        "crop_image_embedding":    # 多出这一个
                    }
                """
        else:
            inputs = processor(image=image,return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.forward(**inputs,output_hidden_states=True)
                vision_embedding = outputs.hidden_states[0]
                entity["image_embedding"] = vision_embedding
                """
                {
                "type": "caption",
                "image_id": image_name,
                "caption_id": i, # the i-th caption in the caption file
                "caption": caption,
                "image_embedding": # 多出这一行
                }
                """




