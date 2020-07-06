# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional
from collections import defaultdict
from datetime import datetime
import random
import json
import logging
import os
import tarfile
import tempfile
import socket
from itertools import chain
import warnings
import torch.nn.functional as F


import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>"],
}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)


def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    logger.info(
        "extracting archive file {} to temp dir {}".format(
            resolved_archive_file, tempdir
        )
    )
    with tarfile.open(resolved_archive_file, "r:gz") as archive:
        archive.extractall(tempdir)
    return tempdir


def get_dataset(tokenizer, dataset_path, dataset_cache):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = (
        dataset_cache + "_" + type(tokenizer).__name__
    )  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    logdir = os.path.join(
        "runs", current_time + "_" + socket.gethostname() + "_" + model_name
    )
    return logdir


def build_input_from_segments(
    persona, history, reply, tokenizer, lm_labels: bool = False, with_eos: bool = True
):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    # Gets the special token values
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    sequence = (
        [[bos] + list(chain(*persona))]
        + history
        + [reply + ([eos] if with_eos else [])]
    )
    sequence = [sequence[0]] + [
        [speaker2 if (len(sequence) - i) % 2 else speaker1] + s
        for i, s in enumerate(sequence[1:])
    ]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    if len(instance["input_ids"]) >= 512:
        print("input is too big!!!")

    instance["token_type_ids"] = [
        speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s
    ]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = (
            ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        )
    return instance


def pad_dataset(dataset: Dict[str, Any], padding: int = 0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    print("===============================================================")
    print("max_l:  ", str(max_l))
    print("===============================================================")
    # Maximum length must be less than 512 because that's all GPT2 was trained with.
    assert max_l <= 512
    for name in PADDED_INPUTS:
        dataset[name] = [
            x + [padding if name != "lm_labels" else -100] * (max_l - len(x))
            for x in dataset[name]
        ]
    return dataset


def get_data_loaders(
    dataset_path: str,
    dataset_cache: str,
    args_num_candidates: int,
    max_history: int,
    personality_permutations: int,
    distributed: bool,
    train_batch_size: int,
    valid_batch_size: int,
    tokenizer,
):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, dataset_path, dataset_cache)
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    if personality_permutations == -1:
        num_personalities = 1
    else:
        num_personalities = personality_permutations

    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])

        # If training then we set the number of candidates to our desired number
        # Otherwise we keep them all for validation loss.
        if args_num_candidates > 0 and dataset_name == "train":
            num_candidates = min(args_num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(num_personalities):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2 * max_history + 1) :]
                    for j, candidate in enumerate(
                        utterance["candidates"][-num_candidates:]
                    ):
                        lm_labels = bool(j == num_candidates - 1)
                        instance = build_input_from_segments(
                            persona, history, candidate, tokenizer, lm_labels
                        )
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    # num_candidates -1 because the last candidate is the true response by the
                    # data format specification.
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                # FIXME:: commented this out for training without persona
                if personality_permutations > 0:
                    persona = [persona[-1]] + persona[:-1]  # permuted personalities

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(
            dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
        )
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            print(dataset_name + ":  " + input_name)
            if input_name != "mc_labels":
                tensor = tensor.view(
                    (-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:]
                )
            else:
                # FIXME:: THERE IS A LARGER PROBLEM HERE, BUT SOMETIMES THE SIZE OF THIS LABEL
                # ARRAY IS WRONG, I'm just forcing it to the right size here but this
                # doesn't address the fundamental issue.
                print(tensor_datasets[dataset_name][0].shape[0])
                tensor = tensor[: tensor_datasets[dataset_name][0].shape[0]]
            tensor_datasets[dataset_name].append(tensor)

    print([x.shape for x in tensor_datasets["train"]])
    logger.info("Build train and validation dataloaders")
    train_dataset = TensorDataset(*tensor_datasets["train"])
    print([x.shape for x in tensor_datasets["valid"]])
    valid_dataset = TensorDataset(*tensor_datasets["valid"])

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if distributed
        else None
    )
    valid_sampler = (
        torch.utils.data.distributed.DistributedSampler(valid_dataset)
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        shuffle=(not distributed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=valid_batch_size,
        shuffle=False,
    )

    logger.info(
        "Train dataset (Batch, Candidates, Seq length): {}".format(
            train_dataset.tensors[0].shape
        )
    )
    logger.info(
        "Valid dataset (Batch, Candidates, Seq length): {}".format(
            valid_dataset.tensors[0].shape
        )
    )
    return train_loader, valid_loader, train_sampler, valid_sampler


def inference(model, device, tokenizer, batch):
    model.eval()
    with torch.no_grad():
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
        # if we dont send labels to model, it doesnt return losses
        lm_logits, mc_logits, *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
        )
        lm_logits_flat_shifted = (
            lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
        )
        lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
        return (
            (lm_logits_flat_shifted, mc_logits),
            (lm_labels_flat_shifted, mc_labels),
        )


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN
    )  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def top_filtering(
    logits, top_k=0.0, top_p=0.9, threshold=-float("Inf"), filter_value=-float("Inf")
):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert (
        logits.dim() == 1
    )  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(
    personality,
    history,
    tokenizer,
    model,
    top_p: float = 0.9,
    top_k: int = 0,
    no_sample: str = "",
    temperature: float = 0.7,
    device: str = "cpu",
    max_length: int = 25,
    min_length: int = 1,
    current_output: Optional[List[str]] = None
):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(max_length):
        instance = build_input_from_segments(
            personality, history, current_output, tokenizer, with_eos=False
        )

        input_ids = torch.tensor(instance["input_ids"], device=device).unsqueeze(0)
        token_type_ids = torch.tensor(
            instance["token_type_ids"], device=device
        ).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)
        if i < min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn(
                        "Warning: model generating special token with probability 1."
                    )
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def make_one_prediction(
    raw_text: str,
    tokenizer_class,
    model_class,
    model_checkpoint: str,
    personality: str = "",
    device: str = "cpu",
    temperature: float = 0.7,
    max_length: int = 25
):
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    model = model_class.from_pretrained(model_checkpoint)
    model.to(device)
    add_special_tokens_(model, tokenizer)

    history = []
    history.append(tokenizer.encode(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, temperature=temperature, max_length=max_length)
    history.append(out_ids)
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return out_text
