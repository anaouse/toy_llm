import random

import torch
import torch.nn as nn
import torch.optim as optim

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


def make_samples(sentences):
    samples = []
    for s in sentences:
        ids = [char2id[c] for c in s]
        for end in range(1, len(ids)):
            ctx = ids[:end]
            tgt = ids[end]
            samples.append((ctx, tgt))
    return samples


samples = make_samples(sentences)
print(f"训练样本数: {len(samples)}")
for ctx, tgt in samples[:4]:
    print(f"  {''.join(id2char[i] for i in ctx)}  →  {id2char[tgt]}")


# ── 模型 ──────────────────────────────────────────────────────────────────────
class MatrixLMPrefix(nn.Module):
    """
    每个 token 是 d×d 矩阵。
    上文连乘，保留所有中间前缀状态：
        S₁ = M₁
        S₂ = M₁ @ M₂
        ...
        Sₙ = M₁ @ ... @ Mₙ

    用一个可学习的注意力机制对所有前缀加权：
        α = softmax(W_a · [s₁, s₂, ..., sₙ])   其中 sₖ = flatten(Sₖ)
        M_result = Σ αₖ · Sₖ

    最后与词表矩阵做 Frobenius cosine 找最近邻。
    """

    def __init__(self, vocab_size, d=12):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size

        # 每个字符对应一个 d×d 矩阵
        self.embeddings = nn.Parameter(torch.randn(vocab_size, d, d) * 0.1)

        # 前缀注意力：输入 flatten 的前缀矩阵 (d²)，输出标量得分
        self.prefix_attn = nn.Linear(d * d, 1, bias=False)

    def get_all_prefixes(self, ids):
        """
        返回所有前缀的连乘矩阵列表
        ids: list[int], 长度 n
        返回: list of (d,d) tensor, 长度 n
            [M₁, M₁@M₂, M₁@M₂@M₃, ...]
        """
        prefixes = []
        M = self.embeddings[ids[0]]
        prefixes.append(M)
        for i in ids[1:]:
            M = M @ self.embeddings[i]
            prefixes.append(M)
        return prefixes  # n 个 (d,d)

    def forward(self, ctx_ids):
        """
        ctx_ids: list[int]
        返回: (vocab_size,) 相似度得分
        """
        prefixes = self.get_all_prefixes(ctx_ids)  # n 个 (d,d)

        # stack → (n, d, d)
        P = torch.stack(prefixes, dim=0)

        # 计算前缀注意力权重
        # flatten 每个前缀矩阵为 (n, d²)
        P_flat = P.view(len(ctx_ids), -1)  # (n, d²)
        attn_scores = self.prefix_attn(P_flat).squeeze(-1)  # (n,)
        alpha = torch.softmax(attn_scores, dim=0)  # (n,)

        # 加权求和所有前缀矩阵
        # alpha: (n,) → (n,1,1) 广播
        M_result = (alpha.view(-1, 1, 1) * P).sum(dim=0)  # (d,d)

        # 与词表做 Frobenius cosine 相似度
        dots = torch.einsum("ij,vij->v", M_result, self.embeddings)  # (V,)
        norm_result = M_result.norm()
        norm_vocab = self.embeddings.norm(dim=(1, 2))
        scores = dots / (norm_result * norm_vocab + 1e-8)
        return scores, alpha  # 同时返回注意力权重方便观察


# ── 训练 ──────────────────────────────────────────────────────────────────────
d = 12
model = MatrixLMPrefix(vocab_size, d=d)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

epochs = 500
for epoch in range(1, epochs + 1):
    random.shuffle(samples)
    total_loss = 0.0
    for ctx, tgt in samples:
        optimizer.zero_grad()
        scores, _ = model(ctx)
        loss = criterion(scores.unsqueeze(0), torch.tensor([tgt]))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  loss={total_loss / len(samples):.4f}")


# ── 推理 ──────────────────────────────────────────────────────────────────────
def predict(context_str, topk=3):
    model.eval()
    with torch.no_grad():
        ids = [char2id[c] for c in context_str if c in char2id]
        if not ids:
            print("未知字符")
            return
        scores, alpha = model(ids)
        probs = torch.softmax(scores, dim=0)
        topk_vals, topk_ids = probs.topk(topk)

        print(f"\n输入: 「{context_str}」")
        for rank, (p, idx) in enumerate(zip(topk_vals, topk_ids), 1):
            print(f"  Top{rank}: 「{id2char[idx.item()]}」  {p.item() * 100:.1f}%")

        # 打印前缀注意力权重
        print(f"  前缀权重: ", end="")
        for k, (char, a) in enumerate(zip(context_str, alpha)):
            prefix = context_str[: k + 1]
            print(f"[{prefix}]={a.item():.2f}", end="  ")
        print()


print("\n─── 推理测试 ───────────────────────────────")
predict("苹果是")
predict("香蕉是")
predict("天空是")
predict("苹果是红")
predict("大海是")

# ── 对比：各前缀的注意力权重分析 ─────────────────────────────────────────────
print("\n─── 前缀权重分析（验证核心假设）───────────────")
print("期望：「苹果是红」→「色」时，[苹果是红] 前缀权重最大")
predict("苹果是红")
print()
print("期望：「苹果是」→「红」时，[苹果是] 前缀权重最大")
predict("苹果是")
