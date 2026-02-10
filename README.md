# adaptivocab-expand (vocab expansion aligner)

이미 만들어진 **확장(Expanded) 토크나이저**를 기반으로,
기존 Hugging Face Causal LM(예: Qwen3) 체크포인트에 대해

- 토큰 임베딩 크기 확장(resize)
- **공유 토큰(shared token)**의 ID 불일치(shift) 보정(토큰 문자열 기준 row 복사)
- **신규 토큰(new token)**에 대해 AdaptiVocab 방식의 **Exponential Embedding Initialization** 적용
- (옵션) special token id를 tokenizer와 config/generation_config에 동기화
- (옵션) 결과 검증(공유 토큰 row 복사 무결성 / 신규 토큰 init 샘플 검사)

를 수행해 **정렬된(aligned) 모델+토크나이저 디렉터리**를 생성하는 유틸/라이브러리입니다.

> 참고 논문(AdaptiVocab):  
> - *AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation*  
> - Itay Nakash, Nitay Calderon, Eyal Ben David, Elad Hoffer, Roi Reichart (COLM 2025 / arXiv:2503.19693) :contentReference[oaicite:1]{index=1}  
> - 링크:
>   ```text
>   https://arxiv.org/abs/2503.19693
>   ```

---

## AdaptiVocab 논문은 “무엇”이고, “왜” 필요한가?

### 문제의식(왜?)
일반-purpose LLM은 다양한 도메인을 커버하도록 학습되어 있지만, **특정 도메인(니치/전문 영역)** 에서 쓰면 “범용성”이 꼭 필요하지 않을 때가 많습니다. 그런데 자동회귀 디코딩은 **토큰 1개 생성마다 forward pass가 필요**해서, 토큰 수가 곧 지연/비용으로 이어집니다. :contentReference[oaicite:2]{index=2}

AdaptiVocab은 여기서 관점을 바꿉니다:
> “모델을 크게 바꾸기보다, **도메인에 맞춰 vocabulary(토크나이저)를 바꿔서** 토큰 수를 줄이면
> 입력 처리/출력 생성이 빨라진다.”

논문은 실제로 “도메인 특화 n-gram 토큰(n-token)”을 추가/교체해 **토큰 사용량을 25%+ 감소**시키면서 성능을 유지하는 방향을 제안합니다. :contentReference[oaicite:3]{index=3}

---

## AdaptiVocab 파이프라인(논문 전체 흐름)

논문에서 제시한 AdaptiVocab은 크게 4단계입니다. :contentReference[oaicite:4]{index=4}

1) **Vocabulary modification**  
   도메인 코퍼스에서 “토큰 절감(token saving)”이 큰 n-token 후보를 고르고, 기존 토큰 일부를 교체

2) **Tokenization patching**  
   어떤 tokenizer에도 적용 가능하도록 encoding/decoding을 패치(논문은 “기존 토큰열을 n-token으로 치환”)

3) **Embedding initialization**  
   새로 추가된 n-token 임베딩을 기존 토큰 임베딩으로부터 “구조를 반영”하여 초기화

4) **Lightweight adaptation fine-tuning**  
   전체 파라미터가 아니라 embedding 및 일부 레이어만 가볍게 미세조정(단일 GPU로 가능하다고 주장) :contentReference[oaicite:5]{index=5}

---

## 이 라이브러리는 논문 중 “어디”를 구현하나?

이 프로젝트는 논문 전체를 재현하기보다, **3) Embedding initialization + (실전에서 꼭 필요한) 정렬/alignment**에 집중합니다.

### ✅ 구현/제공하는 핵심 기능
- **(A) align(정렬): shared token의 id mismatch 보정**
  - 확장 토크나이저에서 special token이 밀리거나 vocab id가 재배치되면,
    같은 토큰 문자열인데도 id가 달라질 수 있습니다.
  - 이때 “토큰 문자열” 기준으로 base 임베딩 row를 expanded 쪽 올바른 id로 복사해서
    **shifted special token 문제를 실전적으로 해결**합니다.

- **(B) Exponential embedding initialization (논문 §3.3 핵심)**
  - 논문은 “새 n-token은 constituent token 임베딩의 단순 평균보다,
    자동회귀 생성 특성을 반영한 가중치가 좋다”고 주장합니다. :contentReference[oaicite:6]{index=6}
  - 직관:
    - **Input embedding**: n-token 내부에서 “마지막 토큰”이 다음 생성에 더 직접적으로 영향을 주므로 더 강조 :contentReference[oaicite:7]{index=7}
    - **Output embedding(LM head)**: n-token을 한 번에 뽑게 하려면 “첫 토큰”이 더 dominant해야 반복을 줄이고 시작을 유도 :contentReference[oaicite:8]{index=8}
  - 논문 수식(요지): 위치 i에 대해 지수 가중치로 정규화한 가중치 w를 사용 :contentReference[oaicite:9]{index=9}
    - input:  `w_i ∝ exp(+2i)`
    - output: `w_i ∝ exp(-2i)`
    - (본 라이브러리는 일반화된 형태로 `alpha`를 노출하여 `w_i ∝ exp(±alpha·i)`로 구현)

> ⚠️ 참고: 모델이 input/output embedding을 **tied(공유 weight)** 로 쓰면,
> output 초기화를 input과 다르게 줄 수 없습니다(둘이 같은 weight이기 때문).

### ❌ 이 라이브러리가 “의도적으로” 하지 않는 것
- n-token 후보 추출/선정(논문 1단계)
- tokenization patching 알고리즘(논문 2단계)
- 경량 fine-tuning(논문 4단계)

즉, “이미 만들어진 expanded tokenizer(외부에서 준비됨)”를 **모델에 안전하게 붙이는 엔지니어링 레이어**입니다.

---

## 이 라이브러리가 하는 일 / 하지 않는 일

### ✅ 하는 일
1. `model.resize_token_embeddings(len(expanded_tokenizer))`
2. **공유 토큰의 ID mismatch**가 있을 때, 토큰 문자열 기준 base row → expanded row 복사
3. **신규 토큰 임베딩 초기화(Exponential init)**
   - 신규 토큰 단일 토큰 surface를 얻고(decode)
   - 그 surface를 base tokenizer로 분해(encode)
   - 분해된 임베딩을 지수 가중치로 합성하여 신규 임베딩으로 설정
4. (선택) Qwen3 스타일 special token(bos/eos/pad) + generation_config 동기화
5. (선택) 검증(저장된 config 정합성, 공유 토큰 무결성, 신규 토큰 init 샘플 검사)

### ❌ 하지 않는 일
- 확장 토크나이저 생성 자체
- 확장 후 파인튜닝/학습
- 논문 전체 파이프라인 end-to-end 실행

---

## 설치

### 로컬 개발 설치(권장)
```bash
pip install -e .
```
### 일반 설치
```bash
pip install .
```

필요 의존성:
-   torch
-   transformers

## 빠른 시작 (CLI)
```bash
adaptivocab-expand \
  --base-model Qwen/Qwen3-8B \
  --base-tokenizer Qwen/Qwen3-8B \
  --expanded-tokenizer DopeorNope/FFT-expanded-naive \
  --output-dir ./qwen3-8b-fft-aligned \
  --alpha 2.0 \
  --device-map auto \
  --dtype auto \
  --dump-json ./new_token_decomp.json
```

옵션:
- `--alpha`: 지수 가중치 강도(기본 2.0, 논문은 ±2i 형태를 사용)
- `--dtype`: auto|fp16|bf16|fp32
- `--device-map`: 예) auto 권장
- `--dump-json`: 신규 토큰 분해 기록 덤프(optional)
- `--skip-verify`: 검증 생략하고 빌드+저장만 수행

## 빠른 시작 (Python API)

```python
from adaptivocab_expand import expand_and_save, ExpansionOptions

opts = ExpansionOptions(
    alpha=2.0,
    dtype="auto",
    device_map="auto",
    trust_remote_code=False,
    dump_json="./new_token_decomp.json",
    verify=True,
    special_token_strategy="qwen3",  # "none" 가능
)

report = expand_and_save(
    base_model="Qwen/Qwen3-8B",
    base_tokenizer="Qwen/Qwen3-8B",
    expanded_tokenizer="DopeorNope/FFT-expanded-naive",
    output_dir="./qwen3-8b-fft-aligned",
    options=opts,
)

print(report)
```


## 인용(Citation)
```bibtex
@misc{nakash2025adaptivocab,
  title={AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation},
  author={Itay Nakash and Nitay Calderon and Eyal Ben David and Elad Hoffer and Roi Reichart},
  year={2025},
  eprint={2503.19693},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

