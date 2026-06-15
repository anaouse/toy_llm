"""
矩阵连乘语言模型 + < 终止符

用法：
  python train_eos.py --mode train                    # 训练
  python train_eos.py --mode infer                    # 推理
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os

EOS = "<"                      # 终止符

# ── 模型 ──────────────────────────────────────────────────────────────────────

class MatrixLM(nn.Module):
    def __init__(self, vocab_size, d=8):
        super().__init__()
        self.d = d
        self.embeddings = nn.Parameter(
            torch.randn(vocab_size, d, d) * 0.05
        )

    def encode_context(self, ids):
        M = self.embeddings[ids[0]]
        for i in ids[1:]:
            M = M @ self.embeddings[i]
        return M

    def forward(self, ctx_ids):
        M_result = self.encode_context(ctx_ids)
        dots = torch.einsum('ij,vij->v', M_result, self.embeddings)
        norm_result = M_result.norm()
        norm_vocab  = self.embeddings.norm(dim=(1, 2))
        scores = dots / (norm_result * norm_vocab + 1e-8)
        return scores

# ── 保存 / 加载 ──────────────────────────────────────────────────────────────

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

# ── 训练 ──────────────────────────────────────────────────────────────────────

def train(args):
    print(f"读取语料: {args.data}")
    with open(args.data, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # 每句话末尾加上 < 作为终止符
    sentences = [line + EOS for line in lines]
    print(f"句子数: {len(sentences)}  示例: {sentences[0]}")

    # 词表
    chars = sorted(set("".join(sentences)))
    if EOS not in chars:
        chars.append(EOS)
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for c, i in char2id.items()}
    vocab_size = len(chars)
    print(f"词表大小: {vocab_size}  字符: {''.join(chars)}")

    # 训练样本
    samples = []
    for s in sentences:
        ids = [char2id[c] for c in s]
        for end in range(1, len(ids)):
            ctx = ids[:end]
            tgt = ids[end]
            samples.append((ctx, tgt))
    print(f"训练样本数: {len(samples)}")

    model = MatrixLM(vocab_size, d=args.d)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(args.save):
        print(f"发现已有模型，继续训练: {args.save}")
        model, char2id, id2char = load_model(args.save)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(samples)
        total_loss = 0.0
        nan_count = 0
        for ctx, tgt in samples:
            # 限制连乘长度，防数值爆炸
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
        avg = total_loss / max(valid, 1)
        print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}  (nan跳过: {nan_count})")

        if epoch % args.save_every == 0:
            save_model(model, char2id, id2char, args.save)

    save_model(model, char2id, id2char, args.save)

# ── 推理 ──────────────────────────────────────────────────────────────────────

def generate(model, char2id, id2char, prompt, max_new=5, temperature=0.8, max_ctx=20):
    """
    给定 prompt，逐字续写。
    - 如果遇到 < 终止符，立即停止
    - 否则最多续写 max_new 个字符
    """
    model.eval()
    eos_id = char2id.get(EOS, -1)

    ids = [char2id[c] for c in prompt if c in char2id]
    if not ids:
        return "[输入字符均不在词表中]"

    result = prompt
    with torch.no_grad():
        for _ in range(max_new):
            ctx = ids[-max_ctx:]          # 滑动窗口
            scores = model(ctx)
            scores = scores / max(temperature, 1e-6)
            probs = torch.softmax(scores, dim=0)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == eos_id:          # 遇到 < 终止
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

        out = generate(
            model, char2id, id2char, prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            max_ctx=args.max_ctx,
        )
        print(f"续写> {out}\n")

# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["train", "infer"], required=True)
    parser.add_argument("--data",   default="train_data.txt")
    parser.add_argument("--save",   default="model_eos.pt")
    parser.add_argument("--d",      type=int,   default=16,    help="矩阵维度")
    parser.add_argument("--epochs", type=int,   default=500)
    parser.add_argument("--lr",     type=float, default=0.005)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_ctx",    type=int, default=20,  help="前向连乘最大长度")
    parser.add_argument("--max_new",    type=int, default=5,   help="推理最多续写字符数")
    parser.add_argument("--temperature",type=float, default=0.8)
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        infer(args)
