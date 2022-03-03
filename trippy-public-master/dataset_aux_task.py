# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
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

import json


def create_examples(path, maxlen=512):
    examples = []
    with open(path, 'r', encoding='utf-8') as reader:
        cnt = 0
        for line in reader:
            sample = json.loads(line)
            if len(sample['token_id']) > maxlen:
                continue
            cnt += 1
            examples.append(sample)
        print('Loaded {} samples out of {}'.format(len(examples), cnt))
    return examples
