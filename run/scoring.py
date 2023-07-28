
import argparse

import MeCab
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_metric import PyRouge

from src.data import jsonlload
from src.utils import get_logger


parser = argparse.ArgumentParser(prog="train", description="Scoring Table to Text")

parser.add_argument("--candidate-path", type=str, help="inference output file path")


def main(args):
    logger = get_logger("scoring")

    rouge = PyRouge(rouge_n=(1, 2, 4))

    logger.info(f'[+] Load Mecab from "/usr/local/lib/mecab/dic/mecab-ko-dic"')
    tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')

    logger.info(f'[+] Load Dataset')
    references_j_list = jsonlload("resource/data/nikluge-2022-table-test-answer.jsonl")
    references = [j["output"] for j in references_j_list]

    candidate_j_list = jsonlload(args.candidate_path)
    candidate = [j["output"] for j in candidate_j_list]

    logger.info(f'[+] Start POS Tagging')
    for idx, sentences in enumerate(references):
        output = []
        for s in sentences:
            tokenized = []
            for mor in tagger.parse(s.strip()).split("\n"):
                if "\t" in mor:
                    splitted = mor.split("\t")
                    token = splitted[0]
                    tokenized.append(token)
            output.append(tokenized)
        references[idx] = output

    for idx, s in enumerate(candidate):
        tokenized = []
        for mor in tagger.parse(s.strip()).split("\n"):
            if "\t" in mor:
                splitted = mor.split("\t")
                token = splitted[0]
                tokenized.append(token)
        candidate[idx] = tokenized

    smoother = SmoothingFunction()
    bleu_score = 0
    for idx, ref in enumerate(references):
        bleu_score += sentence_bleu(ref, candidate[idx], weights=(1.0, 0, 0, 0), smoothing_function=smoother.method1)
    logger.info(f'BLEU Score\t{bleu_score / len(references)}')

    rouge_score = rouge.evaluate(list(map(lambda cdt: " ".join(cdt), candidate)), \
                                 list(map(lambda refs: [" ".join(ref) for ref in refs], references)))
    logger.info(f'ROUGE Score\t{rouge_score["rouge-1"]["f"]}')


if __name__ == "__main__":
    exit(main(parser.parse_args()))
