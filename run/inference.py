
import argparse
from tqdm import tqdm

import torch
from transformers import BartForConditionalGeneration, AutoTokenizer

from src.data import StoryDataLoader, jsonlload, jsonldump
from src.utils import get_logger


parser = argparse.ArgumentParser(prog="train", description="Inference Table to Text with BART")

parser.add_argument("--model-ckpt-path", type=str, help="Table to Text BART model path")
parser.add_argument("--tokenizer", type=str, required=True, help="huggingface tokenizer path")
parser.add_argument("--output-path", type=str, required=True, help="output tsv file path")
parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
parser.add_argument("--max-seq-len", type=int, default=512, help="max sequence length")
parser.add_argument("--summary-max-seq-len", type=int, default=64, help="summary max sequence length")
parser.add_argument("--num-beams", type=int, default=3, help="beam size")
parser.add_argument("--device", type=str, default="cpu", help="inference device")


def main(args):
    logger = get_logger("inference")

    logger.info(f"[+] Use Device: {args.device}")
    device = torch.device(args.device)


    logger.info(f'[+] Load Tokenizer from "{args.tokenizer}"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Load Dataset')
    dataloader = StoryDataLoader("resource/data/nikluge-sc-2023-test.jsonl", tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_seq_len, mode="infer")

    logger.info(f'[+] Load Model from "{args.model_ckpt_path}"')
    model = BartForConditionalGeneration.from_pretrained(args.model_ckpt_path)
    model.to(device)

    logger.info("[+] Eval mode & Disable gradient")
    model.eval()
    torch.set_grad_enabled(False)

    logger.info("[+] Start Inference")
    total_summary_tokens = []
    for batch in tqdm(dataloader):
        dialoge_input = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        summary_tokens = model.generate(
            dialoge_input,
            attention_mask=attention_mask,
            decoder_start_token_id=tokenizer.bos_token_id,
            max_length=args.summary_max_seq_len,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=args.num_beams,
        )
        total_summary_tokens.extend(summary_tokens.cpu().detach().tolist())

    logger.info("[+] Start Decoding")
    decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tqdm(total_summary_tokens)]
    
    j_list = jsonlload("resource/data/nikluge-sc-2023-test.jsonl")
    for idx, oup in enumerate(decoded):
        j_list[idx]["output"] = oup

    jsonldump(j_list, args.output_path)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
