import os
import tempfile
import argparse
import csv
import random

from sentencepiece import SentencePieceTrainer as sp_trainer
from sentencepiece import SentencePieceProcessor as sp_processor

def tokenize(processor, text: str):
    subwords = processor.EncodeAsPieces(text)
    return [s for s in subwords]


def to_ft_format(records, file_name):
    with open(file_name, "w") as w:
        for text, label in records:
            w.write("__label__{} {}\n".format(label, text))


def train(input_file):
    records = []
    with open(input_file, "r") as r:
        next(r)
        reader = csv.reader(r)
        for row in reader:
            _, text, label = row
            text = text.replace("\n", " ").lower()
            tjsya_count = text.count("ться")
            tsya_count = text.count("тся")
            if (tjsya_count + tsya_count) == 1:
                records.append((text, label))
    random.shuffle(records)
    border = int(0.8 * len(records))
    train = records[:border]
    val = records[border:]

    model_path = "subword_models"
    if False:
        temp = tempfile.NamedTemporaryFile(mode="w", delete=False)
        for text, _ in train:
            temp.write(text + "\n")
        temp.close()
        cmd = "--input={} --model_prefix={} --vocab_size={} --model_type={}".format(
            temp.name,
            model_path,
            50000,
            "bpe"
        )
        sp_trainer.Train(cmd)
        os.unlink(temp.name)

    processor = sp_processor()
    processor.load(model_path + ".model")
    fixed_train = []
    for text, label in train:
        text = " ".join(tokenize(processor, text))
        fixed_train.append((text, label))
    fixed_val = []
    for text, label in val:
        text = " ".join(tokenize(processor, text))
        fixed_val.append((text, label))

    to_ft_format(fixed_train, "grammar_endings_train.txt")
    to_ft_format(fixed_val, "grammar_endings_val.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()
    train(**vars(args))
