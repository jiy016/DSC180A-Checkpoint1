"""
code.py
Main experiment code for Weak-to-Strong Generalization on MATH dataset
"""

import asyncio, random, numpy as np, matplotlib.pyplot as plt
from prm800k.grading.grader import grade_answer
from datasets import load_dataset
from openai import AsyncOpenAI
from dataclasses import dataclass
import json, re

# =============== Config and API setup ==================
from config import CFG
client = AsyncOpenAI(api_key=CFG["api_key"])

# =============== Dataset & helper ==================
@dataclass
class MATHQuestion:
    problem: str
    answer: str
    solution: str
    subject: str
    level: int
    unique_id: str
    def get_prompt(self):
        return f"{self.problem}\n\nPlease enclose your final answer in <answer></answer> tags."
    @staticmethod
    def parse_response_for_answer(response: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        return match.group(1) if match else ""

def load_questions(split="train", limit=200):
    with open(f"prm800k/prm800k/math_splits/{split}.jsonl") as f:
        data = [json.loads(line) for line in f][:limit]
    return [MATHQuestion(**d) for d in data]

def eval_model_answers(dataset, answers):
    return [grade_answer(a, q.answer) for q,a in zip(dataset, answers, strict=True)]

# =============== API utils ==================
async def get_messages_with_few_shot_prompt(few_shot_prompt, prompts, model):
    from time import time
    async def one(prompt):
        msg = few_shot_prompt + [{"role": "user", "content": prompt}]
        start = time()
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_completion_tokens=512, messages=msg)
        print(f"Got {model} response in {time()-start:.2f}s")
        return resp
    return await asyncio.gather(*[one(p) for p in prompts])

def get_few_shot_prompt(pairs):
    msg = []
    for p,r in pairs:
        msg.append({"role": "user", "content": p})
        msg.append({"role": "assistant", "content": r})
    return msg

# =============== Main experiment ==================
async def run_experiment(k=4):
    train, test = load_questions("train"), load_questions("test")
    weak_model, strong_model = CFG["weak_model"], CFG["strong_model"]

    # build gold few-shot
    gold_pairs = [(q.get_prompt(), f"<answer>{q.answer}</answer>") for q in train[:k]]
    gold_fs = get_few_shot_prompt(gold_pairs)

    # weak-labelled few-shot
    train_prompts = [q.get_prompt() for q in train[:k]]
    weak_res = await get_messages_with_few_shot_prompt([], train_prompts, model=weak_model)
    weak_answers = [MATHQuestion.parse_response_for_answer(x.choices[0].message.content) for x in weak_res]
    weak_fs = get_few_shot_prompt(list(zip(train_prompts, weak_answers)))

    async def accuracy_on(data, fs, model):
        prompts = [q.get_prompt() for q in data[:CFG["n_test"]]]
        res = await get_messages_with_few_shot_prompt(fs, prompts, model)
        preds = [MATHQuestion.parse_response_for_answer(x.choices[0].message.content) for x in res]
        return float(np.mean(eval_model_answers(data[:CFG["n_test"]], preds)))

    acc_w = await accuracy_on(test, gold_fs, weak_model)
    acc_sw = await accuracy_on(test, weak_fs, strong_model)
    acc_sg = await accuracy_on(test, gold_fs, strong_model)

    denom = acc_sg - acc_w
    pgr = (acc_sw - acc_w) / denom if denom > 1e-9 else np.nan
    print(f"\n[k={k}] Weak+GoldFS={acc_w:.3f}, Strong+WeakFS={acc_sw:.3f}, Strong+GoldFS={acc_sg:.3f}, PGR={pgr:.3f}")

    labels = ["Weak+GoldFS","Strong+WeakFS","Strong+GoldFS"]
    vals = [acc_w, acc_sw, acc_sg]
    plt.bar(labels, vals)
    plt.ylim(0,1); plt.title(f"Few-shot Comparison (k={k})  PGR={pgr:.2f}")
    plt.ylabel("Accuracy"); plt.show()

if __name__ == "__main__":
    asyncio.run(run_experiment(k=CFG["k"]))
