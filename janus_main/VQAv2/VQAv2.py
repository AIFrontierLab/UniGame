# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VQA v2 loading script."""


import csv
import json
from multiprocessing.sharedctypes import Value
import os
from pathlib import Path
import datasets


_CITATION = """\
@InProceedings{VQA,
author = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
title = {VQA: Visual Question Answering},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2015},
} 
"""

_DESCRIPTION = """\
VQA is a new dataset containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer.
"""

_HOMEPAGE = "https://visualqa.org"

_LICENSE = "CC BY 4.0"  # TODO need to credit both ms coco and vqa authors!

_URLS = {
    "questions": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "test-dev": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
        "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    },
    "annotations": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    },
    "images": {
        "train": "http://images.cocodataset.org/zips/train2014.zip",
        "val": "http://images.cocodataset.org/zips/val2014.zip",
        "test-dev": "http://images.cocodataset.org/zips/test2015.zip",
        "test": "http://images.cocodataset.org/zips/test2015.zip",
    },
}
_SUB_FOLDER_OR_FILE_NAME = {
    "questions": {
        "train": "v2_OpenEnded_mscoco_train2014_questions.json",
        "val": "v2_OpenEnded_mscoco_val2014_questions.json",
        "test-dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        "test": "v2_OpenEnded_mscoco_test2015_questions.json",
    },
    "annotations": {
        "train": "v2_mscoco_train2014_annotations.json",
        "val": "v2_mscoco_val2014_annotations.json",
    },
    "images": {
        "train": "train2014",
        "val": "val2014",
        "test-dev": "test2015",
        "test": "test2015",
    },
}


class VQAv2Dataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    # BUILDER_CONFIGS = [
    #     datasets.BuilderConfig(name="v2", version=VERSION, description="TODO later"),
    #     datasets.BuilderConfig(name="v1", version=VERSION, description="TODO later"),
    # ]

    def _info(self):
        features = datasets.Features(
            {
                "question_type": datasets.Value("string"),
                "multiple_choice_answer": datasets.Value("string"),
                "answers": [
                    {
                        "answer": datasets.Value("string"),
                        "answer_confidence": datasets.Value("string"),
                        "answer_id": datasets.Value("int64"),
                    }
                ],
                "image_id": datasets.Value("int64"),
                "answer_type": datasets.Value("string"),
                "question_id": datasets.Value("int64"),
                "question": datasets.Value("string"),
                "image": datasets.Image(),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # urls = _URLS[self.config.name] # TODO later
        data_dir = dl_manager.download_and_extract(_URLS)
        gen_kwargs = {
            split_name: {
                f"{dir_name}_path": Path(data_dir[dir_name][split_name])
                / _SUB_FOLDER_OR_FILE_NAME[dir_name][split_name]
                if split_name in data_dir[dir_name]
                else None
                for dir_name in _URLS.keys()
            }
            for split_name in ["train", "val", "test-dev", "test"]
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=gen_kwargs["train"],
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=gen_kwargs["val"],
            ),
            datasets.SplitGenerator(
                name="testdev",
                gen_kwargs=gen_kwargs["test-dev"],
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs=gen_kwargs["test"],
            ),
        ]

    def _generate_examples(self, questions_path, annotations_path, images_path):
        questions = json.load(open(questions_path, "r"))

        if annotations_path is not None:
            dataset = json.load(open(annotations_path, "r"))

            qa = {ann["question_id"]: [] for ann in dataset["annotations"]}
            for ann in dataset["annotations"]:
                qa[ann["question_id"]] = ann

            for question in questions["questions"]:
                annotation = qa[question["question_id"]]
                # some checks
                assert len(set(question.keys()) ^ set(["image_id", "question", "question_id"])) == 0
                assert (
                    len(
                        set(annotation.keys())
                        ^ set(
                            [
                                "question_type",
                                "multiple_choice_answer",
                                "answers",
                                "image_id",
                                "answer_type",
                                "question_id",
                            ]
                        )
                    )
                    == 0
                )
                record = question
                record.update(annotation)
                record["image"] = str(images_path / f"COCO_{images_path.name}_{record['image_id']:0>12}.jpg")
                yield question["question_id"], record
        else:
            # No annotations for the test split
            for question in questions["questions"]:
                question.update(
                    {
                        "question_type": None,
                        "multiple_choice_answer": None,
                        "answers": None,
                        "answer_type": None,
                    }
                )
                question["image"] = str(images_path / f"COCO_{images_path.name}_{question['image_id']:0>12}.jpg")
                yield question["question_id"], question
