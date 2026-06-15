import torch
import torch.nn as nn
import torch.optim as optim
import random

# ── 语料 ──────────────────────────────────────────────────────────────────────
sentences = [
    "苹果是红色",
    "香蕉是黄色",
    "天空是蓝色",
    "草地是绿色",
    "雪是白色",
    "煤是黑色",
    "玫瑰是红色",
    "柠檬是黄色",
    "大海是蓝色",
    "树叶是绿色",
]

# ── 词表 ──────────────────────────────────────────────────────────────────────
chars = sorted(set("".join(sentences)))
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for c, i in char2id.items()}
vocab_size = len(chars)
print(f"词表大小: {vocab_size}  字符: {''.join(chars)}")

# ── 训练样本：输入前N字，预测第N+1字 ─────────────────────────────────────────
def make_samples(sentences):
    samples = []
    for s in sentences:
        ids = [char2id[c] for c in s]
        for end in range(1, len(ids)):
            ctx = ids[:end]          # 上文
            tgt = ids[end]           # 目标
            samples.append((ctx, tgt))
    return samples

samples = make_samples(sentences)
print(f"训练样本数: {len(samples)}")
for ctx, tgt in samples[:4]:
    print(f"  {''.join(id2char[i] for i in ctx)}  →  {id2char[tgt]}")

# ── 模型：每个字符是一个 d×d 矩阵，上文连乘后与词表矩阵比相似度 ───────────────
class MatrixLM(nn.Module):
    def __init__(self, vocab_size, d=8):
        super().__init__()
        self.d = d
        # 每个字符对应一个 d×d 矩阵，随机初始化
        self.embeddings = nn.Parameter(
            torch.randn(vocab_size, d, d) * 0.1
        )

    def encode_context(self, ids):
        """
        连乘上文中所有字符的矩阵
        ids: list[int]，长度 >= 1
        返回: (d, d) 结果矩阵
        """
        M = self.embeddings[ids[0]]          # 起点
        for i in ids[1:]:
            M = M @ self.embeddings[i]       # 矩阵连乘
        return M

    def forward(self, ctx_ids):
        """
        ctx_ids: list[int]
        返回: (vocab_size,) 相似度得分（未归一化）
        """
        M_result = self.encode_context(ctx_ids)   # (d, d)

        # 与词表每个矩阵算 Frobenius 内积（归一化）
        # embeddings: (V, d, d)
        dots = torch.einsum('ij,vij->v', M_result, self.embeddings)  # (V,)

        norm_result = M_result.norm()
        norm_vocab  = self.embeddings.norm(dim=(1, 2))                # (V,)
        scores = dots / (norm_result * norm_vocab + 1e-8)             # cosine
        return scores

# ── 训练 ──────────────────────────────────────────────────────────────────────
d = 12
model = MatrixLM(vocab_size, d=d)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

epochs = 500
for epoch in range(1, epochs + 1):
    random.shuffle(samples)
    total_loss = 0.0
    for ctx, tgt in samples:
        optimizer.zero_grad()
        scores = model(ctx)                         # (V,)
        loss = criterion(scores.unsqueeze(0),       # (1, V)
                         torch.tensor([tgt]))       # (1,)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  loss={total_loss/len(samples):.4f}")

# ── 推理 ──────────────────────────────────────────────────────────────────────
def predict(context_str, topk=3):
    model.eval()
    with torch.no_grad():
        ids = [char2id[c] for c in context_str if c in char2id]
        if not ids:
            print("未知字符")
            return
        scores = model(ids)
        probs  = torch.softmax(scores, dim=0)
        topk_vals, topk_ids = probs.topk(topk)
        print(f"\n输入: 「{context_str}」")
        for rank, (p, idx) in enumerate(zip(topk_vals, topk_ids), 1):
            print(f"  Top{rank}: 「{id2char[idx.item()]}」  {p.item()*100:.1f}%")

print("\n─── 推理测试 ───────────────────────────────")
predict("苹果是")
predict("香蕉是")
predict("天空是")
predict("苹果是红")
predict("大海是")

# ── 观察"是"的矩阵是否稳定 ────────────────────────────────────────────────────
print("\n─── [是] 的矩阵（训练后）───────────────────")
shi_id = char2id["是"]
M_shi = model.embeddings[shi_id].detach()
print(M_shi.numpy().round(3))
