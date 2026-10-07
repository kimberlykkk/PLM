
from typing import List
import inspect

import torch
from rich import print
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def inference(
    model, 
    tokenizer, 
    sentences: List[str],
    device: str,
    batch_size=16,
    max_seq_len=128
    ) -> List[int]:
    """
    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        sentences (List[str]): _description_
        batch_size (int, optional): _description_. Defaults to 16.
        max_seq_len (int, optional): _description_. Defaults to 128.

    Returns:
        List[int]: [laebl1, label2, label3, ...]
    """
    res = []
    for i in range(0, len(sentences), batch_size):
        batch_sentence = sentences[i:i+batch_size]
        ipnuts = tokenizer(
            batch_sentence,
            truncation=True,
            max_length=max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        accepted = inspect.signature(model.forward).parameters
        model_inputs = {
            key: ipnuts[key].to(device)
            for key in ('input_ids', 'token_type_ids', 'attention_mask')
            if key in ipnuts and key in accepted
        }
        output = model(**model_inputs).logits
        output = torch.argmax(output, dim=-1).cpu().tolist()
        res.extend(output)
    return res


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    saved_model_path = 'checkpoints/comment_classify/model_best'       # 训练模型存放地址
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path) 
    model.to(device).eval()

    sentences = [
        '外表跟图片不太像，而且号码偏大一些，穿着宽松而且裤腿不是很长，除了穿着暖和外还凑合吧，可能我个人试着不太合适',
        '不好看，不值这个价钱',
        '房间超级小，根本就不值688元的价格，特别是在广州这样一个酒店业十分发达的城市，酒店服务差，入住登记时强调要安静的房间。',
    ]
    res = inference(model, tokenizer, sentences, device)
    print('res: ', res)
