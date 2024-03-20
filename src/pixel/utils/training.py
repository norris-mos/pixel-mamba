from dataclasses import field, dataclass
from enum import Enum, auto
from typing import Dict, Optional

import torch
import wandb
from transformers import TrainingArguments

from .misc import format_img, format_img2, format_mask, mark_answer


class Modality(Enum):
    IMAGE = auto()
    TEXT = auto()


@dataclass
class PIXELTrainingArguments(TrainingArguments):
    """
    Custom training arguments that include parameters for early stopping and prediction logging
    """

    early_stopping: Optional[bool] = field(default=True, metadata={"help": "Whether to train with early stopping."})
    early_stopping_patience: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of evaluation steps without increase in validation performance "
            "until training is terminated if training with early stopping."
        },
    )
    log_predictions: Optional[bool] = field(
        default=False, metadata={"help": "Whether to log predictions to file and wandb."}
    )


def debug_log_inputs(inputs: Dict[str, torch.Tensor]):
    """
    Logs inputs as square images to wandb
    Only works when training with Modality.IMAGE
    """

    wandb.init(reinit=False)

    #images = [wandb.Image(format_img(im)) for im in inputs["pixel_values"]]
    nonWandDbImages = format_img(inputs["pixel_values"][0])
    images = wandb.Image(nonWandDbImages)
    #attention_masks = [wandb.Image(format_mask(am)) for am in inputs["attention_mask"]]
    #seq_length = len(inputs["attention_mask"][0])
    #wandb.log(
    #    {
    #        "images": images,
    #        "attention_masks": attention_masks,
    #    }
    #)

    if "patch_mask" in inputs:
        #patch_masks = [wandb.Image(format_mask(pm)) for pm in inputs["patch_mask"]]
        temp_patch_masks = format_mask(inputs["patch_mask"][0])
        patch_masks = wandb.Image(temp_patch_masks)
        wandb.log({"patch_masks": patch_masks, "input": images})#, "images":wandb.Image(nonWandDbImages*temp_patch_masks)})

    if "start_positions" in inputs and "end_positions" in inputs:
        marked_answers = [
            wandb.Image(format_mask(mark_answer(s, e, seq_length)))
            for s, e in zip(inputs["start_positions"], inputs["end_positions"])
        ]
        wandb.log({"answer_spans": marked_answers})

def debug_log_outputs(outputs: Dict[str, torch.Tensor]):
    wandb.init(reinit=False)
    images = wandb.Image(format_img2(outputs["logits"][0]))
   # print(outputs["embedding_output"].shape)
    #masked_images = wandb.Image(format_img2(outputs["embedding_output"][0,1:]))
    #images = [wandb.Image(i) for i in format_img2(outputs["logits"])]
    wandb.log({
        "output_images":images
    })
       # "masked_images":masked_images
    #})

def fineTuning_log_inputs(inputs: Dict[str, torch.Tensor]):
    """
    Logs inputs as square images to wandb
    Only works when training with Modality.IMAGE
    """

    wandb.init(reinit=False)

    nonWandDbImages = format_img(inputs["pixel_values"][0])
    images = wandb.Image(nonWandDbImages)
    wandb.log(
       {
           "images": images
       }
    )


def fineTuning_log_outputs(outputs: Dict[str, torch.Tensor]):
    wandb.init(reinit=False)
    wandb.log({
        "pred":outputs["logits"][0]
    })

    

def format_img3(x: torch.Tensor):
    """
    Wraps an image tensor into square, e.g. from 16x8464 to 368x368 and clips it for proper display
    """
    return clip(unpatchify(x).squeeze())
