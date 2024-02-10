from datasets import interleave_datasets, load_dataset
import sys
import argparse
import numpy as np
sys.path.append('/Users/maxoliverstapyltonnorris/pixel-mamba/src/')
sys.path.append('/Users/maxoliverstapyltonnorris/pixel-mamba/scripts/data/prerendering')
#KMP_DUPLICATE_LIB_OK=TRUE
from PIL import Image
from pixel import PyGameTextRenderer, log_example_while_rendering, push_rendered_chunk_to_hub, get_transforms, PIXELPatchEmbeddings, get_attention_mask
from prerender_wikipedia import process_doc
import json
import matplotlib.pyplot as plt



def RenderAndTransformOneExample():
#####################################################
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renderer_name_or_path",
        type=str,
        help="Path or Huggingface identifier of the text renderer",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to a dataset on disk")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Push data to hub in chunks of N lines",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=-1,
        help="Only look at the first N non-empty lines",
    )
    parser.add_argument("--repo_id", type=str, help="Name of dataset to upload")
    parser.add_argument("--split", type=str, help="Name of dataset split to upload")
    parser.add_argument(
        "--auth_token",
        type=str,
        help="Huggingface auth token with write access to the repo id",
    )

    parsed_args, _ = parser.parse_known_args()

    #####################################################

            
    # Load the dataset in streaming mode
    #dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    dataset = load_dataset('json', data_files='/Users/maxoliverstapyltonnorris/pixel-mamba/notebooks/sample_output.json', split='train')
    #
    text_renderer = PyGameTextRenderer.from_pretrained('Team-PIXEL/pixel-base')


    #####################################################PARAMETERS
    data = {"pixel_values": [], "num_patches": []}
    dataset_stats = {
        "total_uploaded_size": 0,
        "total_dataset_nbytes": 0,
        "total_num_shards": 0,
        "total_num_examples": 0,
        "total_num_words": 0,
    }
    max_pixels = text_renderer.pixels_per_patch * text_renderer.max_seq_length - 2 * text_renderer.pixels_per_patch
    target_seq_length = max_pixels
    idx = 0
    newline_count = 0
    current_doc = ""
    doc_id = 0
    title = "Anarchism"  # Title of the first article in our English wikipedia file
    #####################################################



    # Iterate over each sample in the dataset
    for sample in dataset:
        # Assuming each sample has a 'text' field
        text = sample['text']

        # Process each line in the text
        for line in text.splitlines():
            # Do something with each line
            # For example, print the line
            
            current_doc = line.strip()
            title = 'no title'
            break

    idx, data, dataset_stats = process_doc(
        args=parsed_args,
        text_renderer=text_renderer,
        idx=idx,
        data=data,
        dataset_stats=dataset_stats,
        doc=current_doc,
        target_seq_length=target_seq_length,

    )
    do_normalize=True

    image_mean, image_std = (None, None)
    image_width = text_renderer.pixels_per_patch * text_renderer.max_seq_length


    transforms = get_transforms(
        do_resize=True,
        size=(16, image_width),
        do_normalize=False,
        image_mean=image_mean,
        image_std=image_std,
        do_binary=False,
        
    )
    
    data['transformed_pixel_values'] = [transforms(image) for image in data['pixel_values']]
    data['attention_mask'] = [get_attention_mask(num_patches) for num_patches in data['num_patches']]
    # print(data['attention_mask'][0])
    data['transformed_pixel_values'][0] = data['transformed_pixel_values'][0].unsqueeze(0)
    data['attention_mask'][0] = data['attention_mask'][0].unsqueeze(0)
    return data


#data['transformed_pixel_values'] = [transforms(image) for image in data['pixel_values']]
#numpy_array = data['transformed_pixel_values'][0].numpy()
#
# image = Image.fromarray(data['transformed_pixel_values'])
# image.save("output_image.png")
# print(type(data['pixel_values'][0]))
# print(f"the shape of the data is {data['pixel_values'][0].shape}")
# print(f"the mean value of the pixels is {data['pixel_values'][0].mean()}")
# print(f"the max value of the  pixels is {data['pixel_values'][0].max()}")



#print(f"the mean value of the transformed pixels is {data['transformed_pixel_values'][0].mean()}")
#print(f"the max value of the transformed pixels is {data['transformed_pixel_values'][0].max()}")
#patch_embeddings = PIXELPatchEmbeddings(data['transformed_pixel_values'])
############################################################ This is where the transform occurs


