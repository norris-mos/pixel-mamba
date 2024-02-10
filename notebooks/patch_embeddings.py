# This script gives an example of how patch embeddings are created in the pixel codebase
from datasets import interleave_datasets, load_dataset
import sys
import argparse
sys.path.append('/Users/maxoliverstapyltonnorris/pixel-mamba/src/')
sys.path.append('/Users/maxoliverstapyltonnorris/pixel-mamba/scripts/data/prerendering')
#KMP_DUPLICATE_LIB_OK=TRUE
from PIL import Image
from pixel import PyGameTextRenderer, log_example_while_rendering, push_rendered_chunk_to_hub, get_transforms, PIXELPatchEmbeddings, PIXELEmbeddings,PIXELModel,PIXELConfig
from prerender_wikipedia import process_doc
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
import base64
from PIL import Image
import io
import numpy as np
import torch

def image_to_base64(png_image_file):
    # Convert PngImageFile to bytes
    buffered = io.BytesIO()
    png_image_file.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str
# Load the dataset in streaming mode


def base64_to_image(base64_string):
    # Decode the base64 string
    img_data = base64.b64decode(base64_string)
    
    # Convert binary data to image data
    img = Image.open(io.BytesIO(img_data))
    
    return img

def load_example_data(filename):
    with open(filename,'r') as f:
        section_data = json.load(f)

    for item in section_data:
        base64_string = item['pixel_values']
        image = base64_to_image(base64_string)
        item['pixel_values'] = np.array(image)
    return item
    
item = load_example_data('section_data.json')
print(item['pixel_values'].shape)




config_file = '/Users/maxoliverstapyltonnorris/pixel-mamba/configs/training/fp16_apex_bs32.json'
config = PIXELConfig()

PatchEmbeddings = PIXELPatchEmbeddings()

patch =  item['pixel_values'].reshape((1, 16, 8464))
patch =  patch.reshape((1,1, 16, 8464))
reshaped_patch = np.repeat(patch,3,axis=1)
print(reshaped_patch.shape)

# embedding=PatchEmbeddings.forward(reshaped_patch)
# print(embedding.shape)


patch_embedding = PIXELEmbeddings(config)
embedding,attention_mask,mask_ids_restore = patch_embedding.forward(torch.tensor(reshaped_patch,dtype=torch.float32),attention_mask=None,patch_mask=None)
print(embedding)
























# # Define the range you want to download, for example, from 100 to 200
# start_index = 100
# end_index = 150
# filename = 'section_data.json'
# # Fetch the data
# section_data = []
# for i, example in enumerate(dataset):
#     if start_index <= i < end_index:
#         section_data.append(example)
#     elif i >= end_index:
#         break

# # Assuming section_data is your data that includes PngImageFile objects
# for item in section_data:
#     for key, value in item.items():
#         if isinstance(value, Image.Image):  # or your specific condition to identify a PngImageFile
#             item[key] = image_to_base64(value)

# with open(filename, 'w') as f:
#     json.dump(section_data, f, indent=4)

# Now section_data contains the specific section of the dataset
