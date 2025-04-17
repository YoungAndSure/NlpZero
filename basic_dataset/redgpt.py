# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy
import json
import re

id_to_char = {}
char_to_id = {}


def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_data(file_path, seed=1984, add_eos=False):
    if not os.path.exists(file_path):
        print('No file: %s' % file_path)
        return None

    padchar = '_'
    startchar = '@'
    endchar = '#'
    _update_vocab(padchar)
    _update_vocab(startchar)
    _update_vocab(endchar)

    questions, answers = [], []
    max_question_len = 0
    max_answer_len = 0

    pattern = r"(Human:|Assistant:)"
    for line in open(file_path, 'r'):
        data = json.loads(line)
        dialogue_str = data['dialogue'].replace('\n', '')
        parts = re.split(pattern, dialogue_str)
        parts = [p.strip() for p in parts if p.strip()]
        for i in range(1, len(parts), 2):
            role = parts[i-1].replace(":", "")
            content = parts[i]
            if role == 'Human' :
                questions.append(content)
                max_question_len = max(max_question_len, len(content))
            elif role == 'Assistant' :
                answers.append(content)
                max_answer_len = max(max_answer_len, len(content))

    max_question_len += 1 # padchar
    max_answer_len += 2 # startchar and endchar

    # create vocab dict
    for i in range(len(questions)):
        q, a = questions[i], answers[i]
        _update_vocab(q)
        _update_vocab(a)

    # create numpy array, pad is 0
    x = numpy.zeros((len(questions), max_question_len), dtype=int)
    t = numpy.zeros((len(questions), max_answer_len), dtype=int)

    for i, sentence in enumerate(questions):
        x[i][0:len(sentence)] = [char_to_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        t[i][1:len(sentence) + 1] = [char_to_id[c] for c in list(sentence)]
        # add startchar and endchar
        t[i][0] = char_to_id[startchar]
        t[i][len(sentence) + 1] = char_to_id[endchar]

    # shuffle
    indices = numpy.arange(len(x))
    if seed is not None:
        numpy.random.seed(seed)
    numpy.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 10% for validation set
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)


def get_vocab():
    return char_to_id, id_to_char
