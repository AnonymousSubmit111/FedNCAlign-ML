import torch
import os
import json
import re
from collections import defaultdict
from torchvision import transforms
import pickle as pkl
import pdb
from typing import List, Dict
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

from utils.vqa_utils import get_score, target_tensor
from data.image_datasets.cocoimages_dataset import MSCOCOImagesDataset
from data.image_collation import image_collate
from data.dataset_info import OPENI_LABEL_LIST, INC_TASK_OPENI_LABEL_LIST


def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Ensure the line is not empty
                data.append(json.loads(line))
    return data

def extract_number_from_id(id_string):  # Use regular expression to find numeric part of the id
    numeric_id_pattern = re.compile(r'\d+')
    numeric_id_match = numeric_id_pattern.search(id_string)
    if numeric_id_match:
        return numeric_id_match.group(0)
    else:
        return None


class VQADataset(Dataset):
    def __init__(
        self,
        logger,
        data_dir: str,
        json_text_folder: str,
        json_img_folder: str, 
        images_dataset: MSCOCOImagesDataset,
        split: str,
        task_key: str,
        transform=None,
        remove_old_cls_label=False,
        **kwargs
    ):
        """
        Initiates the VQADataset - loads all the questions (and converts to input IDs using the tokenizer, if provided)
        and answers (including converting each to a numeric label, and a score based on occurence from annotators)
        Every item in self.data corresponds to a single QA pair, with a corresponding image

        Args:
        data_dir : path containing VQA questions and annotations. Also contains mapping from each answer in set of possible answers to a numerical label
        images_dataset : instance of MSCOCOImagesDataset, that is used to retrieve the MS-COCO image for each question
        split: either train/val split

        Returns:
        Loads all annotations into self.data, where each item is a single VQA pair
        """

        self.images_dataset = images_dataset
        if transform:
            self.images_dataset.pil_transform = transform
            self.images_dataset.use_albef = True
        
        self.data_dir = data_dir
        self.json_text_folder = json_text_folder
        self.json_img_folder = json_img_folder
        self.split = split
        self.task_key = task_key
        self.remove_old_cls_label = remove_old_cls_label
        
        if "openi" in task_key: 
            self.text_dir = '{0}/OpenI/{1}/'.format(self.data_dir, self.json_text_folder)
            self.image_dir = '{0}/OpenI/{1}/'.format(self.data_dir, self.json_img_folder)
        elif "mimic" in task_key: 
            self.text_dir = '{0}/Mimic_resized/{1}/'.format(self.data_dir, self.json_text_folder)
            self.image_dir = '{0}/Mimic_resized/{1}/'.format(self.data_dir, self.json_img_folder)
        
        self.tokenizer = kwargs["tokenizer"] if "tokenizer" in kwargs else None
        self.label2idxs = {}

        if ("openi" in task_key) or ("mimic" in task_key):
            self.num_labels = 15
             
        client_id = task_key.split('_')[-1]
        inc_step = task_key.split('_')[-3]
        slipt_id = split.split('_')[0]
        self.client_id = client_id
        self.inc_step = inc_step

        if slipt_id == 'train' or slipt_id == 'val':
            json_file_name = '{0}_step_{1}_client_{2}.json'.format(slipt_id, inc_step, client_id)
        elif slipt_id == 'test':
            json_file_name = '{0}_step_{1}.json'.format(slipt_id, inc_step)
        self.text_data_file = os.path.join(self.text_dir, json_file_name)

        if os.path.isfile(self.text_data_file):
            if ("openi" not in task_key) and ("mimic" not in task_key):
                p = self.text_data_file.replace('.', '_fed.')
            else:
                p = self.text_data_file
                
            if task_key in ["gqa", "vizwiz", 'abstract', 'vqarad', 'slake', 'vqamed2021', 'vqamed2020', 'vqamed2019']:
                self.data = pkl.load(open(p, "rb"))
                for d in self.data:
                    if "question_input_ids" not in d.keys():
                        d["question_input_ids"] = []
            else:
                with open(p, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
        else:
            # Create map from question id to question
            # vqav2 & abstractqq
            # questions = json.load(open(self.questions_file))['questions']
            questions = json.load(open(self.questions_file))
            qid2qdata = {x["question_id"]: x for x in questions}

            # Create data for each annotation
            # vqav2 & abstract
            # annotations = json.load(open(self.annotations_file))['annotations']
            annotations = json.load(open(self.annotations_file))
            self.data = []
            # annotations_dict = {x['question_id']: x for x in annotations}
            # for ques in questions:
            #   qid = ques['question_id']
            #   image_id = int(ques['image'].split('/')[-1].split('.')[0].split('_')[-1])
            #   anno = annotations_dict[qid]
            #   assert image_id == anno['image_id']
            for anno in annotations:
                qid = anno["question_id"]
                # vqav2 & abstract
                # image_id = int(anno['image'].split('/')[-1].split('.')[0].split('_')[-1])
                # pvqa
                image_id = anno["image"].split("/")[-1].split(".")[0]
                # image_id = anno['image'].strip('.jpg').split('/')[-1]
                # image_id = int(anno['image'].strip('.jpg').split('-')[0])

                # Retrieve the question for this annotation
                qdata = qid2qdata[qid]
                # assert qdata['image_id'] == image_id
                # qdata_img_id = int(qdata['image'].split('/')[-1].split('.')[0].split('_')[-1])
                # pvqa
                qdata_img_id = qdata["image"].split("/")[-1].split(".")[0]
                # qdata_img_id = qdata['image'].strip('.jpg').split('/')[-1]
                # qdata_img_id = int(qdata['image'].strip('.jpg').split('-')[0])
                assert qdata_img_id == image_id
                question = qdata["question"]
                if self.tokenizer is not None:
                    tokens = self.tokenizer.tokenize(question)
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    tokens = []
                    input_ids = []

                # Map from each crowdsourced answer to occurrences in annotation
                # answers = [a['answer'] for a in anno['answers']]
                answers = anno["answer"]
                answer_count = defaultdict(int)
                for ans in answers:
                    answer_count[ans] += 1

                # Get label and score (0.3/0.6/1) corresponding to each crowdsourced answer
                labels = []
                scores = []
                answers = []
                for answer in answer_count:
                    if answer not in self.ans2label:
                        continue
                    labels.append(self.ans2label[answer])
                    if task_key in ["toronto", "pvqa", "med", "art", "gqa"] or "clova" in task_key:
                        score = 1 / answer_count[answer]
                    else:
                        score = get_score(answer_count[answer])
                    scores.append(score)
                    answers.append(answer)
                correct_answer = answers[0]

                # Store pre-processed example
                example = {
                    "question_id": qid,
                    "image_id": image_id,
                    "question": question,
                    "question_input_ids": input_ids,
                    "correct_answer": correct_answer,
                    "labels": labels,
                    "answers": answers,
                    "scores": scores,
                }
                
            if not os.path.isdir(self.text_data_file):
                os.makedirs(self.text_data_file)
            pkl.dump(self.data, open(self.text_data_file, "wb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
        index : index of element in self.data to return as data instance

        Returns:
        dictionary containing inputs and targets for model to do VQA
        """        
        example  = self.data[index]
        question_id = example["id"]
        input_ids= example["id"]
        question = example["text"]  # report 
        
        raw_labels = example["label"]  # Get the label string
        text_labels = [label.strip().strip("'") for label in raw_labels.split(',')]  # Strip the outer single quotes and split by commas

        image_id = example["img"].split('/')[-1]
        if "openi" in self.task_key:
            image_path = os.path.join(self.image_dir, image_id)
        elif "mimic" in self.task_key:
            image_id = image_id.split('.')[0]
            image_id_sub_dir = Path(os.path.join(self.image_dir, image_id))
            items = [item.name for item in image_id_sub_dir.iterdir()]
            image_path = os.path.join(image_id_sub_dir, items[0])  # use the first image for each report
        else:
            image_path = os.path.join(self.image_dir, image_id)

        try:
            image = self.images_dataset.get_image_data(image_path)
        except FileNotFoundError:
            print(f"File not found for image_path: {image_path}. Skipping this item.")
            return None

        filtered_labels = []
        for text_item in text_labels:
            if self.remove_old_cls_label:
                if text_item in INC_TASK_OPENI_LABEL_LIST[int(self.inc_step)]:
                    for index in range(len(OPENI_LABEL_LIST)):
                        if text_item == OPENI_LABEL_LIST[index]:
                            filtered_labels.append(index)
                            break
            else:
                for index in range(len(OPENI_LABEL_LIST)):
                    if text_item == OPENI_LABEL_LIST[index]:
                        filtered_labels.append(index)
                        break

        scores = [1.0]
        target_scores = target_tensor(self.num_labels, filtered_labels, scores) # one-hot label
        
        return {
            "question": question,
            "input_ids": input_ids,
            "image": image,
            "labels": filtered_labels,
            "target_scores": target_scores,
            "question_id": question_id,
        }

      
def vqa_batch_collate(batch: List[Dict], visual_input_type: str):
    """
    Collates each model input for all batch items into a single model input (e.g. converts a list of input_ids into a matrix of size (batch_size, max_len))

    Args:
    batch - list of batch items, each item being a dictionary returned by Dataset's __getitem__ method
    visual_input_type: string which specifies the type of visual input

    Returns:
    Dictionary containing batched inputs and outputs
    """

    pad_token = 0  # tokenizer.pad_token_id
    batch = [x for x in batch if x is not None]

    # Pad the text inputs
    #print(batch)
    questions = [x["question"] for x in batch if x is not None]
    input_ids = [x["input_ids"] for x in batch if x is not None]
    #print(input_ids)
    max_len = max([len(x) for x in input_ids], default=0)
    input_ids_padded = []
    attn_masks = []
    for i in range(len(input_ids)):
        # Ensure input_ids[i] is a list of integers
        if isinstance(input_ids[i], str):
           # input_ids_list = list(map(int, input_ids[i].split()))
            input_ids_list = list(map(int, re.findall(r'\d+', input_ids[i])))
    
        else:
            input_ids_list = input_ids[i]

        ids_padded = input_ids_list + [pad_token] * (max_len - len(input_ids_list))
        attn_mask = [1] * len(input_ids_list) + [0] * (max_len - len(input_ids_list))

        input_ids_padded.append(ids_padded)
        attn_masks.append(attn_mask)

    input_ids = torch.tensor(input_ids_padded, dtype=torch.long)
    attn_mask = torch.tensor(attn_masks, dtype=torch.long)

    # Stack the target tensors
    batch_labels = [x["labels"] for x in batch if x is not None]
    batch_scores = [x["target_scores"] for x in batch if x is not None]
    batch_scores = torch.stack(batch_scores, dim=0)

    # Depending on the visual_input_type variable, process the images accordingly
    images = [x["image"] for x in batch if x is not None]
    images = image_collate(images, visual_input_type)

    return {
        "raw_texts": questions,
        "input_ids": input_ids,
        "attn_mask": attn_mask,
        "images": images,
        "target_scores": batch_scores,
        "labels": batch_labels,
    }


def pre_question(question, max_ques_words):
    question = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            question.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
    )
    question = question.rstrip(" ")

    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])

    return question


def vqa_collate_fn_eval(batch):
    # this function is used for ALBEF
    image_list, question_list, answer_list = [], [], []
    
    for image, question, answer in batch:
        image_list.append(image)
        question_list.append(question)
        answer_list.append(torch.tensor(answer))  # Convert to tensor if not already
    
    # Pad the answer tensors to the same length
    padded_answers = pad_sequence(answer_list, batch_first=True, padding_value=-1)
    
    return [
        torch.stack(image_list, dim=0),
        question_list,
        padded_answers,
    ]


def vqa_collate_fn(batch):
    # this function is used for ALBEF
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer #[1,2,3] [2,3,4] [1,2,3,2,3,4]
        n.append(len(answer))
    return [
        torch.stack(image_list, dim=0),
        question_list,
        answer_list,
        torch.Tensor(weight_list),
        n,
    ]


def build_vqa_vilt_dataloader(
    logger, args, images_dataset: MSCOCOImagesDataset, split: str, task_key: str, visual_input_type: str, client_id=-1, **kwargs
) -> torch.utils.data.DataLoader:

    if "train" in split:
        batch_size = args.batch_size
        shuffle = True 
    else:
        batch_size = args.batch_size
        shuffle = False
    
    dataset = VQADataset(logger, args.data_dir, args.json_text_folder, args.json_img_folder, images_dataset, split, task_key, 
                         client_id=client_id, remove_old_cls_label=args.remove_old_cls_label, **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=shuffle,
                                             collate_fn=lambda x: vqa_batch_collate(x, visual_input_type))
    return dataloader, dataset


def build_vqa_albef_dataloader(
    logger, args, data_dir, images_dataset, split: str, task_key: str, client_id=-1, **kwargs
) -> torch.utils.data.DataLoader:

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if "train" in split:
        transform = train_transform
        shuffle = True
        drop_last = True
        collate_fn = vqa_collate_fn
        batch_size = args.batch_size
    else:
        transform = test_transform
        shuffle = False
        drop_last = False
        collate_fn = vqa_collate_fn_eval
        batch_size = args.val_batch_size

    dataset = VQADataset(logger, data_dir, images_dataset, split, task_key, transform, client_id=client_id)

    if torch.distributed.get_rank() == 0:
        logger.info("Created ALBEF VQA {} {} dataloader with len of {}, batch size of {}".format(task_key, split, len(dataset), batch_size))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    return dataloader


