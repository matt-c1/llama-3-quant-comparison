[**Llama 3**](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) is an amazing open large language model.
The 70B variant's weights were published as 130 GB of `bfloat16` tensors in `safetensors` format.
The smaller variant, **8B**, weighs 15 GB.
Thanks to quantization methods, we can run these models on consumer hardware while retaining good quality.
I tested how much quantization affects these models, using [the MMLU test.](https://github.com/hendrycks/test)

# Results

## Quick intro

<details> <summary>What's MMLU?</summary>

The "Massive Multitask Language Understanding" test is composed of 14042 multiple choice questions, non-uniformly distributed among 57 categories. "Correctness" in this article refers to the % of questions the model answered correctly.

<details> <summary>Example question</summary>

Question 45 from the "high school mathematics" category, formatted for Llama 3-Instruct:

```js
<|start_header_id|>user<|end_header_id|>

Question:
To place the first paving stone in a path, Alex starts at the crate of stones, walks three feet, places the stone, and returns to the crate. For each subsequent stone, Alex walks two feet farther each way. Alex will place the first 50 stones in a path. After returning to the crate from placing the $50^\text{th}$ stone, what is the total distance Alex walked, in feet?

Choices:
A: 100
B: 90950
C: 5200
D: 50<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Answer:
```

To which a model is expected to reply with a single token, saying ` A`, ` B`, ` C`, or ` D`. Here, C is correct.

</details>

<details> <summary>Question count per category</summary>

  | | Question Count | Category |
| --:| --:|:-- |
|  1. |  100 | abstract algebra |
|  2. |  135 | anatomy |
|  3. |  152 | astronomy |
|  4. |  100 | business ethics |
|  5. |  265 | clinical knowledge |
|  6. |  144 | college biology |
|  7. |  100 | college chemistry |
|  8. |  100 | college computer science |
|  9. |  100 | college mathematics |
| 10. |  173 | college medicine |
| 11. |  102 | college physics |
| 12. |  100 | computer security |
| 13. |  235 | conceptual physics |
| 14. |  114 | econometrics |
| 15. |  145 | electrical engineering |
| 16. |  378 | elementary mathematics |
| 17. |  126 | formal logic |
| 18. |  100 | global facts |
| 19. |  310 | high school biology |
| 20. |  203 | high school chemistry |
| 21. |  100 | high school computer science |
| 22. |  165 | high school european history |
| 23. |  198 | high school geography |
| 24. |  193 | high school government and politics |
| 25. |  390 | high school macroeconomics |
| 26. |  270 | high school mathematics |
| 27. |  238 | high school microeconomics |
| 28. |  151 | high school physics |
| 29. |  545 | high school psychology |
| 30. |  216 | high school statistics |
| 31. |  204 | high school us history |
| 32. |  237 | high school world history |
| 33. |  223 | human aging |
| 34. |  131 | human sexuality |
| 35. |  121 | international law |
| 36. |  108 | jurisprudence |
| 37. |  163 | logical fallacies |
| 38. |  112 | machine learning |
| 39. |  103 | management |
| 40. |  234 | marketing |
| 41. |  100 | medical genetics |
| 42. |  783 | miscellaneous |
| 43. |  346 | moral disputes |
| 44. |  895 | moral scenarios |
| 45. |  306 | nutrition |
| 46. |  311 | philosophy |
| 47. |  324 | prehistory |
| 48. |  282 | professional accounting |
| 49. | 1534 | professional law |
| 50. |  272 | professional medicine |
| 51. |  612 | professional psychology |
| 52. |  110 | public relations |
| 53. |  245 | security studies |
| 54. |  201 | sociology |
| 55. |  100 | us foreign policy |
| 56. |  166 | virology |
| 57. |  171 | world religions |

</details>

</details>

<details> <summary>What's quantization?</summary>

"Quantizing" a model means converting parts of it to lower precision numerical representations to lower its memory use.
This can allow running large models on limited hardware, but may hurt quality. [Learn more!](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)

</details>

<details> <summary>Bits per weight, bpw?</summary>

Quantization methods typically use mixed precision, expressing different parts of a model in different ways. A way to characterize quantization in one number is to divide its size (or the size of quantized parts of the model) in bits by its number of parameters (weights). Mind that the number of parameters is typically expressed in metric "engineering" units (powers of 1000), and file size in JEDEC units (powers of 1024), so the formula is:

$$
\mathup{bpw} = \left(\frac{1024}{1000}\right)^3 \cdot \frac{\text{(size in GB)}}{\text{(billions of parameters)}}
\approx 1.0737 \ \frac{\text{(size in GB)}}{\text{(billions of parameters)}}
$$

</details>

<details> <summary>EXL2? GGUF?</summary>

These are popular quantized LLM file formats, working with [Exllama v2](https://github.com/turboderp/exllamav2) and [llama.cpp](https://github.com/ggerganov/llama.cpp/), respectively.

</details>

## Correctness vs File Size

The following plot shows how the models slowly lose the ability to answer MMLU questions correctly the more quantized they are.

<img src="./plots/MMLU-Correctness-vs-File-Size.svg">

- The points labeled "70B" correspond to the 70B variant of the Llama 3 model, the rest the 8B variant.
- "gguf" used [files](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF) provided by `bartowski`. The "Q-numbers"
- "exl2" also used [files](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-exl2) provided by `bartowski`, in fp16, 8 bpw (bits per weight), 6.5 bpw, 5 bpw, 4.25 bpw, 3.5 bpw.
- "transformers" refers to evaluating the model using the HuggingFace `transformers` module and its supported `bitsandbytes` quantization-on-load options: 8 bit, 4 bit fp4, 4 bit nf4 (normalized float).

<details> <summary>Data table</summary>

\* Note: the 70B model was evaluated with only 50 questions per category, the 8B with full MMLU.

bpw here was calculated only considering Llama 3's `model.layers.*.weight` layers, as the approach to quantizing the rest of the model differs significantly between methods.

| File size [GB] | MMLU [%] | bpw | Model | Quant | Type |
| --:| --:| --:|:--:|:--:|:--:|
| 46.52 | \* 80.82 |  5.66 | 70B | Q5_K_M | GGUF |
| 35.30 | \* 80.46 |  4.26 | 70B | IQ4_XS | GGUF |
| 29.74 | \* 80.06 |  3.50 | 70B | IQ3_M | GGUF |
| 25.58 | \* 79.09 |  3.04 | 70B | IQ3_XXS | GGUF |
| 22.46 | \* 77.01 |  2.62 | 70B | IQ2_M | GGUF |
| 20.71 | \* 76.05 |  2.38 | 70B | IQ2_S | GGUF |
| 19.69 | \* 74.94 |  2.35 | 70B | IQ2_XS | GGUF |
| 17.79 | \* 72.31 |  2.11 | 70B | IQ2_XXS | GGUF |
| 15.60 | \* 65.21 |  1.81 | 70B | IQ1_M | GGUF |
| 14.97 | 65.20 | 16.00 | 8B | fp16 | GGUF |
| 14.96 | 65.20 | 16.00 | 8B | fp16 | Exl2 |
| 14.96 | 65.21 | 16.00 | 8B | bf16 | transformers |
| 14.29 | \* 61.18 |  1.63 | 70B | IQ1_S | GGUF |
|  7.96 | 65.20 |  7.99 | 8B | 8.00 bpw | Exl2 |
|  7.95 | 65.23 |  8.50 | 8B | Q8_0 | GGUF |
|  7.48 | 64.53 |  8.00 | 8B | 8bit | transformers |
|  6.75 | 64.99 |  6.49 | 8B | 6.50 bpw | Exl2 |
|  6.14 | 65.06 |  6.56 | 8B | Q6_K | GGUF |
|  5.43 | 64.27 |  5.00 | 8B | 5.00 bpw | Exl2 |
|  5.34 | 64.90 |  5.67 | 8B | Q5_K_M | GGUF |
|  5.21 | 64.88 |  5.50 | 8B | Q5_K_S | GGUF |
|  4.82 | 63.36 |  4.25 | 8B | 4.25 bpw | Exl2 |
|  4.58 | 64.64 |  4.82 | 8B | Q4_K_M | GGUF |
|  4.37 | 64.63 |  4.54 | 8B | Q4_K_S | GGUF |
|  4.36 | 64.33 |  4.52 | 8B | IQ4_NL | GGUF |
|  4.21 | 60.28 |  3.50 | 8B | 3.50 bpw | Exl2 |
|  4.14 | 64.39 |  4.28 | 8B | IQ4_XS | GGUF |
|  4.03 | 62.85 |  4.08 | 8B | Q3_K_L | GGUF |
|  3.74 | 62.89 |  3.79 | 8B | Q3_K_M | GGUF |
|  3.74 | 63.42 |  4.00 | 8B | 4bit-nf4 | transformers |
|  3.74 | 61.75 |  4.00 | 8B | 4bit-fp4 | transformers |
|  3.52 | 62.55 |  3.50 | 8B | IQ3_M | GGUF |
|  3.43 | 62.13 |  3.46 | 8B | IQ3_S | GGUF |
|  3.41 | 59.14 |  3.44 | 8B | Q3_K_S | GGUF |
|  3.28 | 61.19 |  3.26 | 8B | IQ3_XS | GGUF |
|  3.05 | 60.52 |  3.04 | 8B | IQ3_XXS | GGUF |
|  2.96 | 55.90 |  2.90 | 8B | Q2_K | GGUF |
|  2.75 | 57.56 |  2.64 | 8B | IQ2_M | GGUF |
|  2.57 | 53.98 |  2.40 | 8B | IQ2_S | GGUF |
|  2.43 | 49.98 |  2.37 | 8B | IQ2_XS | GGUF |
|  2.23 | 43.50 |  2.14 | 8B | IQ2_XXS | GGUF |
|  2.01 | 28.83 |  1.84 | 8B | IQ1_M | GGUF |
|  1.88 | 26.47 |  1.66 | 8B | IQ1_S | GGUF |

</details>

### Key takeaways:
- Modern GGUF "I-Quants" seem to offer the best quality at given size.
- All tested quantization formats perform the same at fp16.
- Going as low as ~5 bpw has minimal impact on quality in this test.
- `transformers` quantization is slightly lower quality, except `bitsandbytes` 4 bit **nf4** which interestingly keeps up with GGUF, beating all "\*Q3\*" quants.

### Is exllamav2 under-performing?
- It *can be* faster than fully offloaded GGUF, depending on the task. In this test it was almost twice as fast, processing 14 thousand tokens per second vs 7500 for llama.cpp.
It seems that for the same bpw, EXL2 resulted in worse MMLU scores. But for now ExLlamaV2 still offers some unique advantages:
- It offers 4 bit cache, which allows quartering the memory necessary for context size. If you count context, EXL2 may become your best option until llama.cpp implements this feature. Especially if you need 16k+ context length. (Supported by other models, supposedly a long context Llama 3 is in training.)
- I personally found it to be easiest and most pleasant to work with from Python's level, but that's subjective and depends on the task.

## Confidence vs bpw

**Confidence** here is the average *normalized* probability that the model would give a correct answer *if we only consider the 4 tokens corresponding to valid answers.*
Random noise would result in 25% confidence (and 25% correctness) because I'm normalizing the 4 possible answers to add up to 100%.

<img src="./plots/Confidence-vs-bpw-no-head.svg">

The main takeaway here is that the 70B model is less affected by quantization. Perhaps it's more sparse relative to the 8B one.
Extremely low quants of 70B remain somewhat useable, whereas 8B-IQ1-M and -S are near the random noise threshold.

<img src="./plots/Confidence-loss-vs-bpw.svg">

Here I plotted the loss of confidence (change from maximum). It seems to change like $\propto \text{bpw}^{-4.25}$.
I had to include this, because no one can resist "things looking linear on a log-log plot."

# Methodology

## Shortcomings of this test

- To calculate MMLU, I made a mistake of using the first five questions in each "test" category for 5-shot evaluation, instead of using the "dev" set. This changes the MMLU values slightly, but isn't an issue in this study, as I'm only comparing my own tests to one another and focusing on their relative differences.
- I skipped around 20 questions where the 5-shot prompt was above 2048 tokens.

## Applicability of this test

From anecdotal experience, it seems that quantization affects "rigorous" tasks like writing working source code more than open-ended tasks like creative writing. It would be interesting to methodically measure the effect of quantization on demanding programming benchmarks.

## Shortcomings of MMLU

### It's *okay* for this purpose

For this study, MMLU is fine because it's an ablation study. Even if MMLU is a flawed quality benchmark, it's *good enough* to see how a model's answers change with quantization.

### Is MMLU still relevant?

It *used to be* arguable whether MMLU is a good benchmark, but it does not really matter any more, as top models are scoring around 90%. There's not much room for improvement, but fortunately [harder benchmarks](https://lmsys.org/blog/2024-04-19-arena-hard/) are being proposed.

### It's partially broken

Some of MMLU questions are [broken, arguably useful, opinionated, or lack necessary context.](https://derenrich.medium.com/errors-in-the-mmlu-the-deep-learning-benchmark-is-wrong-surprisingly-often-7258bb045859)
Example, question 133 from the "high school psychology" category:

> As a result of an accident, Abdul lost sight in his right eye. To judge the distance of vehicles when he is driving, Abdul is able to rely on cues of
> 
> A. I only  
> B. II only  
> C. III only  
> D. I and II only  

This question lacks statements numbered I, II, and III, necessary to answer it.

## Inference code

I based my code on the test included in [ExLlamaV2's repository,](https://github.com/turboderp/exllamav2/blob/master/tests/test_mmlu.py) but modified it heavily.

You can find pre-compiled Python wheels for inference libraries from the [text-generation-webui repository.](https://github.com/oobabooga/text-generation-webui/blob/main/requirements.txt)

The snippets assume that you load a list of strings representing the questions and answers as `prompts` and `answers`.

### transformers

<details> <summary>Simplified transformers source code</summary>

```py
import torch
import transformers

model_path = "path/to/model"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
config = transformers.PretrainedConfig.from_pretrained(model_path)
config.max_position_embeddings = 2048

quantization_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = transformers.LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    config=config,
    device_map="auto",
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
)

answer_tokens = tokenizer.encode(
    " A B C D", add_special_tokens=False, return_tensors="pt"
)

with torch.no_grad(): # crucial for lower memory use
    for prompt, answer in zip(prompts, answers):
        prompt_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )

        logits_ans = model.forward(prompt_ids.cuda()).logits[:, -1, answer_tokens].cpu()
        # process the answer
        torch.cuda.empty_cache()
```

</details>

### llama-cpp-python

Mind to install a correct version of `llama-cpp-python`, with CUDA support if you can use it. Adjust `n_gpu_layers` if you can't offload the full model. A model's total number of layers is listed in its `config.json` as `num_hidden_layers`.

<details> <summary>Simplified llama-cpp-python source code</summary>

```py
import torch
from llama_cpp_cuda_tensorcores import Llama, llama_tokenizer

model_path = "path/to/model.gguf"
tokenizer_base = "path/to/model"  # where tokenizer.json is located

llama_params = {
    "model_path": model_path,
    "n_ctx": 2048,  # Text context, 0 = from model
    "n_batch": 512,  # Prompt processing maximum batch size
    "n_gpu_layers": -1,  # -1 offloads ALL layers
    "n_threads": 8,  # Number of threads to use for generation
    "n_threads_batch": 8,  # Number of threads to use for batch processing
    "logits_all": False,  # Not needed for model.eval()
    "offload_kqv": True,  # Offload K, Q, V to GPU.
    "tokenizer": llama_tokenizer.LlamaHFTokenizer.from_pretrained(
        tokenizer_base
    ),  # Optional tokenizer to override the default tokenizer from llama.cpp.
    "verbose": False,  # Don't print verbose output to stderr.
}

model = Llama(**llama_params)

answer_tokens = model.tokenize(" A B C D".encode(), add_bos=False)

for prompt, answer in zip(prompts, answers):
    prompt_ids = model.tokenize(prompt.encode(), add_bos=False)

    model.reset()
    model.eval(prompt_ids)
    logits = model.scores[model.n_tokens - 1]
    logits_ans = torch.tensor([logits[i] for i in answer_tokens], device="cpu")
```

</details>

### ExLlamaV2

<details> <summary>Simplified ExLlamaV2 source code</summary>

```py
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
)

model_path = "path/to/model-exl2"
config = ExLlamaV2Config()
config.model_dir = model_path
config.prepare()
config.max_seq_len = 2048
model = ExLlamaV2(config)
tokenizer = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache(model, max_seq_len=2048, lazy=True)
model.load_autosplit(cache)

answer_logits = tokenizer.encode(" A B C D")

for prompt, answer in zip(prompts, answers):
    prompt_ids = tokenizer.encode(prompt)
    logits = model.forward(prompt_ids, last_id_only=True)
    logits_ans = logits[:, :, answer_logits].cpu()
```

</details>
