# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
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

import os
import json

import dataset_woz2
import dataset_sim
import dataset_multiwoz21
import dataset_aux_task


class DataProcessor(object):
    def __init__(self, dataset_config):
        with open(dataset_config, "r", encoding='utf-8') as f:
            raw_config = json.load(f)
        self.class_types = raw_config['class_types']
        self.slot_list = raw_config['slots']
        self.label_maps = raw_config['label_maps']

    def get_train_examples(self, data_dir, **args):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, **args):
        raise NotImplementedError()

    def get_test_examples(self, data_dir, **args):
        raise NotImplementedError()


class Woz2Processor(DataProcessor):
    def get_train_examples(self, data_dir, args):
        return dataset_woz2.create_examples(os.path.join(data_dir, 'woz_train_en.json'),
                                            'train', self.slot_list, self.label_maps, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_woz2.create_examples(os.path.join(data_dir, 'woz_validate_en.json'),
                                            'dev', self.slot_list, self.label_maps, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_woz2.create_examples(os.path.join(data_dir, 'woz_test_en.json'),
                                            'test', self.slot_list, self.label_maps, **args)


class Multiwoz21Processor(DataProcessor):
    def get_train_examples(self, data_dir, is_fewshot, args):
        fewshot_config = "_fewshot" if is_fewshot else ""
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, f'train_dials{fewshot_config}.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'train', self.slot_list, self.label_maps, **args)
    def get_dev_examples(self, data_dir, is_fewshot, args):
        fewshot_config = "_fewshot" if is_fewshot else ""
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, f'val_dials{fewshot_config}.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'train', self.slot_list, self.label_maps, **args)
    def get_test_examples(self, data_dir, is_fewshot, args):
        fewshot_config = "_fewshot" if is_fewshot else ""
        return dataset_multiwoz21.create_examples(os.path.join(data_dir, f'test_dials{fewshot_config}.json'),
                                                  os.path.join(data_dir, 'dialogue_acts.json'),
                                                  'train', self.slot_list, self.label_maps, **args)

class SimProcessor(DataProcessor):
    def get_train_examples(self, data_dir, args):
        return dataset_sim.create_examples(os.path.join(data_dir, 'train.json'),
                                           'train', self.slot_list, **args)

    def get_dev_examples(self, data_dir, args):
        return dataset_sim.create_examples(os.path.join(data_dir, 'dev.json'),
                                           'dev', self.slot_list, **args)

    def get_test_examples(self, data_dir, args):
        return dataset_sim.create_examples(os.path.join(data_dir, 'test.json'),
                                           'test', self.slot_list, **args)


class AuxTaskProcessor(object):
    def get_aux_task_examples(self, data_dir, data_name, max_seq_length):
        file_path = os.path.join(data_dir, '{}_train.json'.format(data_name))
        return dataset_aux_task.create_examples(file_path, max_seq_length)


PROCESSORS = {"woz2": Woz2Processor,
              "sim-m": SimProcessor,
              "sim-r": SimProcessor,
              "multiwoz21": Multiwoz21Processor,
              "multiwoz23": Multiwoz21Processor,
              "aux_task": AuxTaskProcessor}
