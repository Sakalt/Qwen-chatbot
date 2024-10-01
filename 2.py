from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# モデルとトークナイザーの読み込み
model_name = "Sakalti/qwen2.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# チャット関数の定義
def チャットボット(user_input, history=[]):
    # ユーザーの入力と履歴を結合
    history.append(f"ユーザー: {user_input}")
    input_text = "\n".join(history) + "\nボット:"

    # トークナイズ
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # モデルによる出力生成
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id)

    # 出力のデコード
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ボットの応答を取り出し、履歴に追加
    bot_response = output_text.split("ボット:")[-1].strip()
    history.append(f"ボット: {bot_response}")

    return bot_response, history

# チャットボットの実行
history = []
while True:
    user_input = input("あなた: ")
    if user_input.lower() in ["終了", "やめる"]:
        break
    response, history = チャットボット(user_input, history)
    print(f"ボット: {response}")
