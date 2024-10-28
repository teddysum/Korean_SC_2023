# 이야기 완성 Baseline
본 레포지토리는  '2024년 국립국어원 인공지능의 한국어 능력 평가' 상시 과제 중 '이야기 완성'에 대한 베이스라인 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.


학습, 추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.

### Baseline
|Model|ROUGE-1|
|:---:|---|
|KoBART (v2)|0.322|
<!--
|Model|ROUGE-1|BLUE|BLUERT|BERT Score|
|:---:|---|---|---|---|
|KoBART (v2)|0.322|0.328|0.386|0.762|
-->
평가 코드 : https://github.com/teddysum/korean_evaluation.git

## Directory Structure
```
# 학습에 필요한 리소스들이 들어있습니다.
resource
├── data
└── tokenizer

# 실행 가능한 python 스크립트가 들어있습니다.
run
├── infernece.py
└── train.py

# 학습에 사용될 커스텀 함수들이 구현되어 있습니다.
src
├── data.py     # torch dataloader
├── module.py   # pytorch-lightning module
└── utils.py
```

## Data Format
```
{
    "id": "nikluge-2023-sc-dev-000001",
    "input": {
        "sentence1": "그는 거실에서 영화를 보다 잠들었다.",
        "sentence3": "그는 형이 깨워줘서 겨우 방에 들어가 다시 잤다."
    },
    "output": "그 모습을 본 형이 그를 깨웠다."
}
```

### Enviroments
Docker Image
```
docker pull nvcr.io/nvidia/pytorch:22.08-py3 
```

Docker Run Script
```
docker run -dit --gpus all --shm-size=8G --name baseline_sc nvcr.io/nvidia/pytorch:22.08-py3
```

Install Python Dependency
```
pip install -r requirements.txt
```

Install MeCab
```
./install_mecab.sh
```

## How to Run
### Train
```
python -m run train \
    --output-dir outputs/sc \
    --model-path gogamza/kobart-base-v2 \
    --tokenizer gogamza/kobart-base-v2 \
    --gpus 0 1 --epoch 5 \
    --max-learning-rate 2e-5 --min-learning-rate 1e-6 \
    --warmup-rate 0.1 --r3f-lambda 0.1 \
    --max-seq-len 512 \
    --batch-size-train 12 --batch-size-valid 12 \
    --logging-interval 100 --evaluate-interval 1.0 \
    --seed 93 --wandb-project SC
```

### Inference
```
python -m run inference \
    --model-ckpt-path outputs/sc/<your-model-ckpt-path> \
    --tokenizer gogamza/kobart-base-v2 \
    --output-path test_output.jsonl \
    --batch-size=16 \
    --max-seq-len 512 \
    --summary-max-seq-len 256 \
    --num-beams 5 \
    --device cuda:1
```

## Reference
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
transformers (https://github.com/huggingface/transformers)  
KoBART (v2) (https://huggingface.co/gogamza/kobart-base-v2)  
