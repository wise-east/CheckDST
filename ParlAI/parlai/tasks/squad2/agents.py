#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.teachers import FixedDialogTeacher, DialogTeacher
from parlai.utils.io import PathManager
from .build import build

import json
import os
import random


def add_common_cmdline_args(parser):
    agent = parser.add_argument_group("Squad2 teacher arguments")
    agent.add_argument(
        "--impossible-answer-string",
        type=str,
        default="",
        help='Set the label for impossible answers; defaults to an empty string, but one might try something like "I do not know"',
    )

    agent.add_argument(
        "--use_prompts",
        type=bool,
        default=True,
        help="add natural text instructions for the DST task.",
    )


class IndexTeacher(FixedDialogTeacher):
    """
    Hand-written SQuAD teacher, which loads the json squad data and implements its own
    `act()` method for interacting with student agent, rather than inheriting from the
    core Dialog Teacher. This code is here as an example of rolling your own without
    inheritance.

    This teacher also provides access to the "answer_start" indices that specify the
    location of the answer in the context.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        add_common_cmdline_args(parser)
        return parser

    def __init__(self, opt, shared=None):
        build(opt)
        super().__init__(opt, shared)

        if self.datatype.startswith("train"):
            suffix = "train"
        else:
            suffix = "dev"
        datapath = os.path.join(opt["datapath"], "SQuAD2", suffix + "-v2.0.json")
        self.data = self._setup_data(datapath)

        self.id = "squad2"
        self.reset()

    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article["paragraphs"][paragraph_idx]
        qa = paragraph["qas"][qa_idx]
        question = qa["question"]
        answers = []
        answer_starts = []
        if not qa["is_impossible"]:
            for a in qa["answers"]:
                answers.append(a["text"])
                answer_starts.append(a["answer_start"])
        else:
            answers = [self.opt["impossible_answer_string"]]
            answers = ["No answer"]
        context = paragraph["context"]
        plausible = qa.get("plausible_answers", [])

        action = {
            "id": "squad",
            "text": context
            + "\n"
            + question
            + "Say not answerable if it can't be answered with the given content.",
            "labels": answers,
            "plausible_answers": plausible,
            "episode_done": True,
            "answer_starts": answer_starts,
        }
        return action

    def _setup_data(self, path):
        with PathManager.open(path) as data_file:
            self.squad = json.load(data_file)["data"]
        self.examples = []

        for article_idx in range(len(self.squad)):
            article = self.squad[article_idx]
            for paragraph_idx in range(len(article["paragraphs"])):
                paragraph = article["paragraphs"][paragraph_idx]
                num_questions = len(paragraph["qas"])
                for qa_idx in range(num_questions):
                    self.examples.append((article_idx, paragraph_idx, qa_idx))


class DefaultTeacher(DialogTeacher):
    """
    This version of SQuAD inherits from the core Dialog Teacher, which just requires it
    to define an iterator over its data `setup_data` in order to inherit basic metrics,
    a default `act` function.

    For SQuAD, this does not efficiently store the paragraphs in memory.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        add_common_cmdline_args(parser)
        return parser

    def __init__(self, opt, shared=None):
        self.datatype = opt["datatype"]
        build(opt)
        if opt["datatype"].startswith("train"):
            suffix = "train"
        else:
            suffix = "dev"
        opt["datafile"] = os.path.join(opt["datapath"], "SQuAD2", suffix + "-v2.0.json")
        self.id = "squad2"
        super().__init__(opt, shared)

    def format_content(self, context, question, title, answer):
        # from T0 work ref: https://arxiv.org/pdf/2110.08207.pdf

        question_options = [
            "Answer the question depending on the context.",
            "What is the answer?",
        ]

        question_options2 = [
            "Can you tell me ",
            "Please tell me ",
            "Tell me ",
            "From the passage, ",
            "I want to know ",
            "I want to ask ",
            "What is the answer to: ",
            "Find the answer to: ",
            "Answer: ",
            "",
        ]

        topic_question_options = [
            "What is this about? ",
            "What is the paragraph about? ",
            "Get the topic from: ",
            "From the passage, get the topic ",
            "I want to know the topic. ",
            "Topic from the passage: ",
            "Topic from the paragraph: ",
            "What is the following passage about?",
        ]

        topic_question_options2 = [
            "This is about ",
            "What is this about? ",
            "The paragraph is about",
            "What is the paragraph about? ",
            "Get the topic: ",
            "From the passage, the topic is",
            "I want to know the topic. ",
            "Topic from the passage: ",
            "Topic from the paragraph: ",
        ]

        topic_question_options3 = [
            "Determine the topic of the question-answer pair. ",
            "Find the topic. ",
            "What is the topic from this? ",
        ]

        title = title.replace("_", " ")
        if answer[0] == self.opt["impossible_answer_string"]:
            answer = ["Answer not in context"]
            templates = []
        else:
            templates = [
                (
                    f"Determine the question that you might have asked to get back the following answer for the given context\n Context: {context}; Answer: {answer}; Question: ",
                    question,
                ),
                (
                    f"What is a question that would give the following answer? Answer: {answer}; Question: ",
                    question,
                ),
                (
                    f"{random.choice(topic_question_options3)} Question: {question}; Answer: {answer}; Topic: ",
                    title,
                ),
            ]

        templates += [
            (
                f"{random.choice(question_options)}\n Context: {context}; Question: {question}; Answer: ",
                answer,
            ),
            (f"{context} {random.choice(question_options2)} {question}", answer),
            (f"{random.choice(topic_question_options)} {context}", title),
            (f"{context} {random.choice(topic_question_options2)}", title),
        ]

        formatted_context, answer = random.choice(templates)
        if isinstance(answer, str):
            answer = [answer]
        return formatted_context, answer

    def setup_data(self, path):
        print("loading: " + path)
        with PathManager.open(path) as data_file:
            self.squad = json.load(data_file)["data"]
        for article in self.squad:
            title = article["title"]
            # each paragraph is a context for the attached questions
            for paragraph in article["paragraphs"]:
                # each question is an example
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    ans_list = [{"text": self.opt["impossible_answer_string"]}]
                    if not qa["is_impossible"]:
                        ans_list = qa["answers"]
                    context = paragraph["context"]
                    body = context + "\n" + question
                    # use the formatted versions
                    answers = tuple(a["text"] for a in ans_list)
                    if self.opt.get("use_prompts", True):
                        body, answers = self.format_content(
                            context, question, title, answers
                        )
                    yield (body, answers), True


class OpenSquadTeacher(DialogTeacher):
    """
    This version of SQuAD inherits from the core Dialog Teacher, which just requires it
    to define an iterator over its data `setup_data` in order to inherit basic metrics,
    a default `act` function.

    Note: This teacher omits the context paragraph
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        add_common_cmdline_args(parser)
        return parser

    def __init__(self, opt, shared=None):
        self.datatype = opt["datatype"]
        build(opt)
        if opt["datatype"].startswith("train"):
            suffix = "train"
        else:
            suffix = "dev"
        opt["datafile"] = os.path.join(opt["datapath"], "SQuAD2", suffix + "-v2.0.json")
        self.id = "squad2"
        super().__init__(opt, shared)

    def setup_data(self, path):
        print("loading: " + path)
        with PathManager.open(path) as data_file:
            self.squad = json.load(data_file)["data"]
        for article in self.squad:
            # each paragraph is a context for the attached questions
            for paragraph in article["paragraphs"]:
                # each question is an example
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    ans_iter = [{"text": self.opt["impossible_answer_string"]}]
                    if not qa["is_impossible"]:
                        ans_iter = qa["answers"]
                    answers = [a["text"] for a in ans_iter]
                    yield (question, answers), True


class TitleTeacher(DefaultTeacher):
    """
    This version of SquAD inherits from the Default Teacher.

    The only
    difference is that the 'text' field of an observation will contain
    the title of the article separated by a newline from the paragraph and the
    query.
    Note: The title will contain underscores, as it is the part of the link for
    the Wikipedia page; i.e., the article is at the site:
    https://en.wikipedia.org/wiki/{TITLE}
    Depending on your task, you may wish to remove underscores.
    """

    def __init__(self, opt, shared=None):
        self.id = "squad_title"
        build(opt)
        super().__init__(opt, shared)

    def setup_data(self, path):
        print("loading: " + path)
        with PathManager.open(path) as data_file:
            self.squad = json.load(data_file)["data"]
        for article in self.squad:
            title = article["title"]
            # each paragraph is a context for the attached questions
            for paragraph in article["paragraphs"]:
                # each question is an example
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    ans_iter = [{"text": self.opt["impossible_answer_string"]}]
                    if not qa["is_impossible"]:
                        ans_iter = qa["answers"]
                    answers = [a["text"] for a in ans_iter]
                    context = paragraph["context"]
                    yield ("\n".join([title, context, question]), answers), True


class SentenceIndexTeacher(IndexTeacher):
    """
    Index teacher where the labels are the sentences the contain the true answer.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        try:
            import nltk
        except ImportError:
            raise ImportError("Please install nltk (e.g. pip install nltk).")
        # nltk-specific setup
        st_path = "tokenizers/punkt/{0}.pickle".format("english")
        try:
            self.sent_tok = nltk.data.load(st_path)
        except LookupError:
            nltk.download("punkt")
            self.sent_tok = nltk.data.load(st_path)

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article["paragraphs"][paragraph_idx]
        qa = paragraph["qas"][qa_idx]
        context = paragraph["context"]
        question = qa["question"]

        answers = []
        if not qa["is_impossible"]:
            answers = [a["text"] for a in qa["answers"]]

        # temporarily remove '.', '?', '!' from answers for proper sentence
        # tokenization
        edited_answers = []
        for answer in answers:
            new_answer = answer.replace(".", "").replace("?", "").replace("!", "")
            context = context.replace(answer, new_answer)
            edited_answers.append(new_answer)

        edited_sentences = self.sent_tok.tokenize(context)
        sentences = []

        for sentence in edited_sentences:
            for i in range(len(edited_answers)):
                sentence = sentence.replace(edited_answers[i], answers[i])
                sentences.append(sentence)

        for i in range(len(edited_answers)):
            context = context.replace(edited_answers[i], answers[i])

        labels = []
        label_starts = []
        for sentence in sentences:
            for answer in answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    label_starts.append(context.index(sentence))
                    break
        if len(labels) == 0:
            labels.append("")

        plausible = []
        if qa["is_impossible"]:
            plausible = qa["plausible_answers"]
            labels = [self.opt["impossible_answer_string"]]

        action = {
            "id": "squad",
            "text": context + "\n" + question,
            "labels": labels,
            "plausible_answers": plausible,
            "episode_done": True,
            "answer_starts": label_starts,
        }
        return action


class SentenceIndexEditTeacher(SentenceIndexTeacher):
    """
    Index teacher where the labels are the sentences the contain the true answer.

    Some punctuation may be removed from the context and the answer for tokenization
    purposes.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article["paragraphs"][paragraph_idx]
        qa = paragraph["qas"][qa_idx]
        context = paragraph["context"]
        question = qa["question"]

        answers = [""]
        if not qa["is_impossible"]:
            answers = [a["text"] for a in qa["answers"]]

        # remove '.', '?', '!' from answers for proper sentence
        # tokenization
        edited_answers = []
        for answer in answers:
            new_answer = answer.replace(".", "").replace("?", "").replace("!", "")
            context = context.replace(answer, new_answer)
            edited_answers.append(new_answer)

        edited_sentences = self.sent_tok.tokenize(context)

        labels = []
        label_starts = []
        for sentence in edited_sentences:
            for answer in edited_answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    label_starts.append(context.index(sentence))
                    break

        plausible = []
        if qa["is_impossible"]:
            plausible = qa["plausible_answers"]
            labels = [self.opt["impossible_answer_string"]]

        action = {
            "id": "squad",
            "text": context + "\n" + question,
            "labels": labels,
            "plausible_answers": plausible,
            "episode_done": True,
            "answer_starts": label_starts,
        }
        return action


class SentenceLabelsTeacher(IndexTeacher):
    """
    Teacher which contains the question as the text, the sentences as the label
    candidates, and the label as the sentence containing the answer.

    Some punctuation may be removed for tokenization purposes.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        try:
            import nltk
        except ImportError:
            raise ImportError("Please install nltk (e.g. pip install nltk).")
        # nltk-specific setup
        st_path = "tokenizers/punkt/{0}.pickle".format("english")
        try:
            self.sent_tok = nltk.data.load(st_path)
        except LookupError:
            nltk.download("punkt")
            self.sent_tok = nltk.data.load(st_path)

    def get(self, episode_idx, entry_idx=None):
        article_idx, paragraph_idx, qa_idx = self.examples[episode_idx]
        article = self.squad[article_idx]
        paragraph = article["paragraphs"][paragraph_idx]
        qa = paragraph["qas"][qa_idx]
        context = paragraph["context"]
        question = qa["question"]

        answers = [""]
        if not qa["is_impossible"]:
            answers = [a["text"] for a in qa["answers"]]

        # remove '.', '?', '!' from answers for proper sentence
        # tokenization
        edited_answers = []
        for answer in answers:
            new_answer = answer.replace(".", "").replace("?", "").replace("!", "")
            context = context.replace(answer, new_answer)
            edited_answers.append(new_answer)

        edited_sentences = self.sent_tok.tokenize(context)

        labels = []
        for sentence in edited_sentences:
            for answer in edited_answers:
                if answer in sentence and sentence not in labels:
                    labels.append(sentence)
                    break

        plausible = []
        if qa["is_impossible"]:
            plausible = qa["plausible_answers"]
            labels = [self.opt["impossible_answer_string"]]
        action = {
            "id": "SquadSentenceLabels",
            "text": question,
            "labels": labels,
            "plausible_answers": plausible,
            "label_candidates": edited_sentences,
            "episode_done": True,
        }

        return action
