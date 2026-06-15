"""
矩阵连乘 + 前缀加权语言模型
用法：
  训练：  python model.py --mode train --data corpus.txt --save model.pt
  推理：  python model.py --mode infer --save model.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import random

EOS = "<EOS>"

# ─────────────────────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────────────────────

def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return [line + EOS for line in lines]

def build_vocab(sentences):
    chars = sorted(set("".join(sentences)))
    if EOS not in chars:
        chars.append(EOS)
    char2id = {c: i for i, c in enumerate(chars)}
    id2char  = {i: c for c, i in char2id.items()}
    return char2id, id2char

def make_samples(sentences, char2id, window=100):
    stream = []
    for s in sentences:
        stream += [char2id[c] for c in s if c in char2id]

    samples = []
    step = window // 2
    for start in range(0, len(stream) - 1, step):
        chunk = stream[start: start + window]
        for i in range(1, len(chunk)):
            ctx = chunk[:i]
            tgt = chunk[i]
            samples.append((ctx, tgt))
    return samples

# ─────────────────────────────────────────────────────────────
# 模型
# ─────────────────────────────────────────────────────────────

class MatrixLM(nn.Module):
    def __init__(self, vocab_size, d=16):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.embeddings = nn.Parameter(
            torch.randn(vocab_size, d, d) * 0.05
        )
        self.prefix_attn = nn.Linear(d * d, 1, bias=False)

    def forward(self, ctx_ids):
        M = self.embeddings[ctx_ids[0]]
        prefixes = [M]
        for i in ctx_ids[1:]:
            M = M @ self.embeddings[i]
            prefixes.append(M)

        P = torch.stack(prefixes, dim=0)
        alpha = torch.softmax(
            self.prefix_attn(P.view(len(ctx_ids), -1)).squeeze(-1), dim=0
        )
        M_result = (alpha.view(-1, 1, 1) * P).sum(0)

        dots = torch.einsum('ij,vij->v', M_result, self.embeddings)
        norm_r = M_result.norm()
        norm_v = self.embeddings.norm(dim=(1, 2))
        return dots / (norm_r * norm_v + 1e-8)

# ─────────────────────────────────────────────────────────────
# 保存 / 加载
# ─────────────────────────────────────────────────────────────

def save_model(model, char2id, id2char, path):
    torch.save({
        "state_dict": model.state_dict(),
        "char2id": char2id,
        "id2char": id2char,
        "d": model.d,
        "vocab_size": model.vocab_size,
    }, path)
    print(f"模型已保存 -> {path}")

def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MatrixLM(ckpt["vocab_size"], ckpt["d"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt["char2id"], ckpt["id2char"]

# ─────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────

def train(args):
    print(f"读取语料: {args.data}")
    sentences = load_corpus(args.data)
    print(f"句子数: {len(sentences)}")

    char2id, id2char = build_vocab(sentences)
    vocab_size = len(char2id)
    print(f"词表大小: {vocab_size}")

    print("构建训练样本（滑动窗口）...")
    samples = make_samples(sentences, char2id, window=args.window)
    print(f"训练样本数: {len(samples)}")

    model = MatrixLM(vocab_size, d=args.d)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(args.save):
        print(f"发现已有模型，继续训练: {args.save}")
        model, char2id, id2char = load_model(args.save)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        total_loss = 0.0
        nan_count  = 0

        for ctx, tgt in samples:
            # 限制连乘长度，防止长序列数值爆炸
            ctx = ctx[-args.max_ctx:]

            optimizer.zero_grad()
            scores = model(ctx)
            loss = criterion(scores.unsqueeze(0), torch.tensor([tgt]))

            if torch.isnan(loss):
                nan_count += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # 归一化 embedding，防止奇异值连乘后爆炸
            with torch.no_grad():
                norms = model.embeddings.norm(dim=(1, 2), keepdim=True)
                model.embeddings.data /= (norms + 1e-8)

            total_loss += loss.item()

        valid = len(samples) - nan_count
        avg   = total_loss / max(valid, 1)
        print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}  (nan跳过: {nan_count})")

        if epoch % args.save_every == 0:
            save_model(model, char2id, id2char, args.save)

    save_model(model, char2id, id2char, args.save)

# ─────────────────────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────────────────────

def generate(model, char2id, id2char, prompt, max_new=100, temperature=0.8, max_ctx=20):
    model.eval()
    eos_id = char2id.get(EOS, -1)

    ids = [char2id[c] for c in prompt if c in char2id]
    if not ids:
        return "[输入字符均不在词表中]"

    result = prompt
    with torch.no_grad():
        for _ in range(max_new):
            ctx = ids[-max_ctx:]   # 滑动窗口
            scores = model(ctx)
            scores = scores / max(temperature, 1e-6)
            probs  = torch.softmax(scores, dim=0)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == eos_id:
                break

            next_char = id2char[next_id]
            result += next_char
            ids.append(next_id)

    return result

def infer(args):
    if not os.path.exists(args.save):
        print(f"找不到模型文件: {args.save}，请先训练。")
        return

    model, char2id, id2char = load_model(args.save)
    print(f"模型加载完毕  词表: {model.vocab_size}  矩阵维度: {model.d}x{model.d}")
    print("输入提示字符后回车续写，quit 退出。\n")

    while True:
        try:
            prompt = input("输入> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if prompt.lower() == "quit":
            break
        if not prompt:
            continue

        output = generate(
            model, char2id, id2char, prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            max_ctx=args.max_ctx,
        )
        print(f"续写> {output}\n")

# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        choices=["train", "infer"], required=True)
    parser.add_argument("--data",        default="corpus.txt")
    parser.add_argument("--save",        default="model.pt")
    parser.add_argument("--d",           type=int,   default=16,    help="矩阵维度")
    parser.add_argument("--window",      type=int,   default=100,   help="滑动窗口大小")
    parser.add_argument("--max_ctx",     type=int,   default=20,    help="前向连乘最大长度")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=0.005)
    parser.add_argument("--save_every",  type=int,   default=5)
    parser.add_argument("--max_new",     type=int,   default=80,    help="推理最多生成字符数")
    parser.add_argument("--temperature", type=float, default=0.8,   help="采样温度")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        infer(args)
