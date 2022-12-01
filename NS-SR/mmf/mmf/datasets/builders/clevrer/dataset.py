import json
import os

import numpy as np
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import is_master, synchronize
from mmf.utils.general import get_mmf_root
from mmf.utils.text import VocabFromText, tokenize
from PIL import Image


_CONSTANTS = {
    "question_and_answer_folder": "question_and_answer",
    "dataset_key": "clevrer",
    "empty_folder_error": "CLEVRER dataset folder is empty.",
    "question_key": "question",
    "answer_key": "answer",
    "train_dataset_key": "train",
    "frames_folder": "frames",
    "vocabs_folder": "vocabs",
}

_TEMPLATES = {
    "data_folder_missing_error": "Data folder {} for CLEVR is not present.",
    "question_json_file": "clevrer_{}_{}.json",

    "vocab_file_template": "{}_{}_{}_vocab.txt",
}


class CLEVRERDataset(BaseDataset):
    """Dataset for CLEVR. CLEVR is a reasoning task where given an image with some
    3D shapes you have to answer basic questions.

    Args:
        dataset_type (str): type of dataset, train|val|test
        config (DictConfig): Configuration Node representing all of the data necessary
                             to initialize CLEVR dataset class
        data_folder: Root folder in which all of the data will be present if passed
                     replaces default based on data_dir and data_folder in config.

    """

    def __init__(self, config, dataset_type, data_folder=None, *args, **kwargs):
        super().__init__(_CONSTANTS["dataset_key"], config, dataset_type)
        self._data_dir = config.data_dir
        self.video_len = config.video_len
        self.anno_dir = self._data_dir + '/annotation/'
        # defined what type of clerver u used
        self._question_type = config.question_type 
        self.multi_choice_type = ['explanatory','predictive','counterfactual'] 
        self.load()

    def load(self):
        self.frame_folder_path = os.path.join(
            self._data_dir, _CONSTANTS["frames_folder"], self._dataset_type
        )

        with open(
            os.path.join(
                self._data_dir,
                _CONSTANTS["question_and_answer_folder"],
                _TEMPLATES["question_json_file"].format(self._dataset_type,self._question_type),
            )
        ) as f:
            self.questions_and_answer = json.load(f)

            # Vocab should only be built in main process, as it will repetition of same task
            if is_master() and self._dataset_type == 'train':
                self._build_vocab(self.questions_and_answer, _CONSTANTS["question_key"],self._question_type)
                self._build_vocab(self.questions_and_answer, _CONSTANTS["answer_key"],self._question_type)
            synchronize()

    def __len__(self):
        return len(self.questions_and_answer)

    def _get_vocab_path(self, attribute,_type):
        return os.path.join(
            self._data_dir,
            _CONSTANTS["vocabs_folder"],
            _TEMPLATES["vocab_file_template"].format(self.dataset_name,_type,attribute),
        )

    def _build_vocab(self, questions, attribute, _type):
        # Vocab should only be built from "train" as val and test are not observed in training
        if self._dataset_type != _CONSTANTS["train_dataset_key"]:
            return

        vocab_file = self._get_vocab_path(attribute,_type)

        # Already exists, no need to recreate
        if os.path.exists(vocab_file):
            return

        # Create necessary dirs if not present
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)

        if _type in self.multi_choice_type and attribute == _CONSTANTS["answer_key"]:
            sentences = [choice[attribute] for question in questions for choice in question['choices']]
        else:
            sentences = [question[attribute] for question in questions]
        build_attributes = self.config.build_attributes

        # Regex is default one in tokenize i.e. space
        kwargs = {
            "min_count": build_attributes.get("min_count", 1),
            "keep": build_attributes.get("keep", [";", ","]),
            "remove": build_attributes.get("remove", ["?", "."]),
        }

        if attribute == _CONSTANTS["answer_key"]:
            kwargs["only_unk_extra"] = False

        vocab = VocabFromText(sentences, **kwargs)

        with open(vocab_file, "w") as f:
            f.write("\n".join(vocab.word_list))

    def _load_videos(self,path,video_len):
        frame_list = os.listdir(path).sort()
        video = []
        stride = int(len(frame_list) // video_len)
        for i in range(0,len(frame_list),stride):
            frame_path = path + '/' + frame_list[i]
            frame = np.true_divide(Image.open(frame_path).convert("RGB"), 255)
            frame = frame.astype(np.float32)
            video.append(frame)

        video_tensor = torch.from_numpy(video.transpose(0, 3, 1, 2))
        return video_tensor

    def _get_multi_choice_details(self,current_sample,data):
        question = data["question"]
        question_tokens = tokenize(question, keep=[";", ","], remove=["?", "."])
        question_processed = self.text_processor({"tokens": question_tokens})
        current_sample.question = question_processed["text"]

        choices= data['choices']
        idx = np.random.randint(0,len(choices)-1)
        choice = choices[idx]['choice']
        choice_tokens = tokenize(choice, keep=[";", ","], remove=["?", "."])
        choice_processed = self.text_processor({"tokens": choice_tokens})
        current_sample.choice = choice_processed["text"]
        current_sample.choice_id = choice = choices[idx]['choice_id']
        answer_processed = self.answer_processor({"answers": [choices[idx]['answer']]})
        current_sample.answers = answer_processed["answers"]
        current_sample.targets = answer_processed["answers_scores"]

        return current_sample
        
    def __getitem__(self, idx):
        data = self.questions_and_answer[idx]
        current_sample = Sample()
        # adding annotation
        # video_file = data["video_filename"]
        # annotation_path = self.anno_dir +'annotation_' + video_file.split(".").split('_')[1] + '.json'
        # with open(annotation_path) as f:
        #     annotation = json.load(f)
        #current_sample.annotation = annotation

        video_path = os.path.join(self.frame_folder_path, data["video_filename"].split('.')[0])
        debugging = True
        if debugging:
            current_sample.video = torch.rand(3, 4, 320, 480)
        else:
            current_sample.video = self._load_videos(video_path,self.video_len)
        # for debugging
        DEBUG = True
        if DEBUG:
            current_sample.video = torch.rand(3, 4, 320, 480)
        else:
            video_path = os.path.join(self.frame_folder_path, data["video_filename"].split('.')[0])
            current_sample.video = self._load_videos(video_path,self.video_len) # ,self.stride)

        if self._question_type == 'descriptive':
            question = data["question"]
            tokens = tokenize(question, keep=[";", ","], remove=["?", "."])
            processed = self.text_processor({"tokens": tokens})
            current_sample.question = processed["text"]
            processed = self.answer_processor({"answers": [data["answer"]]})
            current_sample.answers = processed["answers"]
            current_sample.targets = processed["answers_scores"]

        elif self._question_type in self.multi_choice_type:
           current_sample = self._get_multi_choice_details(current_sample, data)
        else:
            raise RuntimeError(
               "No Such Type of Questions"
            )

        return current_sample
#mmf_run config=projects/others/cnn_lstm/clevrer/defaults.yaml datasets=clevrer model=cnn_lstm_for_vqa run_type=train_val
