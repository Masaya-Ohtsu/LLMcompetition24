# 必要なライブラリを読み込み
from unsloth import FastLanguageModel
from peft import PeftModel
import json
from tqdm import tqdm

# 初期設定
HF_TOKEN = "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"
max_seq_length = 1024
dtype = None  # Noneにして自動設定
load_in_4bit = True  # 4bit量子化を有効化

# モデルとLoRAアダプターの設定
model_id = "llm-jp/llm-jp-3-13b"
lora_adapter = "Masaya02/llm-jp-3-13b-it_lora"

# ベースモデルとトークナイザーのロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    trust_remote_code=True,
)

# データセットの読み込み
datasets = []
with open("./elyza-tasks-100-TV_0.jsonl", "r") as f:
    item = ""
    for line in f:
        line = line.strip()
        item += line
        if item.endswith("}"):
            datasets.append(json.loads(item))
            item = ""

model = PeftModel.from_pretrained(model, lora_adapter, token = HF_TOKEN)  # LoRAアダプターを適用

# 推論を実行する準備
FastLanguageModel.for_inference(model)

# 推論処理
results = []
for dt in tqdm(datasets):
    input_text = dt["input"]

    prompt = f"""
### 指示
{input_text}
### 回答"""

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=512, use_cache=True, do_sample=False, repetition_penalty=1.2
    )
    answer_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_output = answer_text.split("### 回答\n")[-1]

    # 結果を保存
    results.append({
        "task_id": dt["task_id"],
        "input": input_text,
        "output": answer_output
    })

# 結果を保存
# jsonlで保存
with open("output.jsonl", 'w', encoding='utf-8') as f:
    for result in results:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

print("推論が完了しました。結果は 'output.jsonl' に保存されています。")