# GM-SWA：算法与训练算子加速说明

## 摘要

长上下文语言建模的核心挑战在于：模型既需要保持对局部上下文的精确建模能力，又需要以可接受的计算与缓存开销处理窗口之外的长期依赖。滑动窗口注意力在效率上具有显著优势，但其固定窗口机制会直接截断超出窗口范围的历史信息，限制模型的长程检索与跨段依赖建模能力。针对这一问题，我们提出 **GM-SWA（Gated Memory Sliding-Window Attention）**。该方法在保留精确局部滑窗注意力的基础上，为每个 key-value 头引入少量可学习的 memory slots，并将被窗口淘汰的 value 通过门控递推写入紧凑记忆状态，使未来 token 在访问局部窗口的同时，也能够读取压缩后的长程上下文表示。

GM-SWA 的核心由两部分组成。其一是算法层面的记忆读写机制：模型从当前隐藏状态预测门控系数，从被滑窗移出的 value 构造记忆更新，并通过逐维 gated interpolation 维护一个小型递归 memory state；在读取阶段，memory state 被映射为额外的 memory keys 与 values，并与局部滑窗分支通过统一归一化进行融合。其二是训练层面的高效实现：由于记忆更新本质上是一个逐 token 的递推过程，若直接串行展开会显著拖慢长序列训练；而基于累积乘积的闭式扫描虽然速度较快，却会在长序列下产生数值不稳定的中间量。为此，我们进一步将该递推重写为仿射变换序列，并构造一个与原始递推严格等价的稳定 affine prefix-scan，实现对 padded 与 packed variable-length 序列的并行训练。

从方法定位上看，GM-SWA 并不试图替代滑动窗口注意力，而是在其上增加一个低容量、低开销、可训练的长期记忆通道，从而在“纯局部注意力”与“高成本全局长上下文机制”之间提供一个更具工程可行性的折中方案。本文档系统整理了 GM-SWA 的算法定义、记忆读写形式、局部分支与记忆分支的融合方式，以及训练阶段稳定并行扫描算子的设计与实现，为后续论文撰写提供精确的方法基础。

这份文档单独整理 **GM-SWA 的算法定义** 与 **训练阶段的算子加速实现**。写法尽量贴近当前代码实现，尤其对应：

- `fla/layers/gated_mem_swa.py`
- `fla/models/gated_mem_swa/modeling_gated_mem_swa.py`

下文默认讨论单层 decoder attention block；多层模型只是按层重复该结构。

---

## 一、GM-SWA 算法

### 1. 记号与基本设定

设输入隐藏状态为

$$
\mathbf{H} = (\mathbf{h}_1,\dots,\mathbf{h}_T), \qquad \mathbf{h}_t \in \mathbb{R}^{d}.
$$

模型有：

- 查询头数 $H$
- key-value 头数 $H_{\mathrm{kv}}$
- 每头维度 $d_h = d / H$
- query-group 数 $G = H / H_{\mathrm{kv}}$
- 滑窗大小 $W$
- 每个 key-value 头的 memory slot 数 $M$

标准 sliding-window attention 只保留最近 $W$ 个 token 的局部 KV cache，因此每个时刻 $t$ 的局部上下文是：

$$
\mathcal{L}_t = \{ \max(1, t-W+1), \dots, t \}.
$$

GM-SWA 在这个局部注意力之外，为每个 key-value 头再维护一个小的 memory state：

$$
\mathbf{M}_t \in \mathbb{R}^{H_{\mathrm{kv}} \times M \times d_h}.
$$

它的目标不是替代局部注意力，而是为**被窗口淘汰的历史信息**提供一个压缩的、可学习的长期通道。

---

### 2. 局部分支

与普通 SWA 一样，首先计算

$$
\mathbf{Q}_t = W_Q \mathbf{h}_t,\qquad
\mathbf{K}_t = W_K \mathbf{h}_t,\qquad
\mathbf{V}_t = W_V \mathbf{h}_t.
$$

随后施加 RoPE，并仅在最近 $W$ 个 token 上计算精确注意力。对第 $h$ 个 query 头，局部分支输出为

$$
\mathbf{o}^{\mathrm{local}}_{t,h}
=
\sum_{j \in \mathcal{L}_t}
\alpha^{\mathrm{local}}_{t,h,j}\,\mathbf{v}_{j,h},
$$

其中

$$
\alpha^{\mathrm{local}}_{t,h,j}
=
\frac{
\exp\left(\langle \mathbf{q}_{t,h}, \mathbf{k}_{j,h}\rangle / \sqrt{d_h}\right)
}{
\sum_{j' \in \mathcal{L}_t}
\exp\left(\langle \mathbf{q}_{t,h}, \mathbf{k}_{j',h}\rangle / \sqrt{d_h}\right)
}.
$$

实现上这部分由 `flash_attn_func` / `flash_attn_varlen_func` 计算，因此局部分支本身已经是高效的。

---

### 3. Memory Write：如何把被淘汰的信息写入记忆

当时刻 $t$ 到来时，窗口左侧被淘汰的 value 向量是第 $t-W$ 个 token 的 value。对每个 key-value 头 $h$ 和每个 memory slot $s$，GM-SWA 会生成一个候选更新：

$$
\mathbf{u}^{(h,s)}_t = \phi^{(h,s)}\!\left(\mathbf{v}^{(h)}_{t-W}\right),
$$

其中 $\phi$ 在实现中有两种模式：

1. `mem_proj_mode = linear`
   - 用线性层把 $\mathbb{R}^{d_h}$ 投影到 $\mathbb{R}^{M d_h}$
   - 再 reshape 成 $M$ 个 slot

2. `mem_proj_mode = scale`
   - 每个 slot 对 evicted value 做逐维缩放

同时，模型根据当前 token 的隐藏状态 $\mathbf{h}_t$ 生成一个 gate：

$$
\mathbf{g}^{(h,s)}_t
=
\sigma\!\left(\psi^{(h,s)}(\mathbf{h}_t)\right),
\qquad
\mathbf{g}^{(h,s)}_t \in (0,1)^{d_h}.
$$

这里 $\psi$ 在实现中也有两种模式：

1. `mem_gate_mode = linear`
   - 用线性层从当前 hidden state 预测 gate

2. `mem_gate_mode = param`
   - gate 直接是可学习参数，与 token 无关

于是 memory state 的写入更新为：

$$
\mathbf{m}^{(h,s)}_t
=
\mathbf{g}^{(h,s)}_t \odot \mathbf{m}^{(h,s)}_{t-1}
+
\left(1-\mathbf{g}^{(h,s)}_t\right)\odot \mathbf{u}^{(h,s)}_t.
$$

这是一个逐维 gated interpolation：

- 当 gate 接近 $1$ 时，保留旧 memory
- 当 gate 接近 $0$ 时，用新 update 覆盖 memory

这一定义非常关键，因为它说明 GM-SWA 的 memory 不是外部检索库，也不是显式 token 历史缓存，而是**每个 KV head 上的小型可学习递归状态**。

---

### 4. 哪些 token 真正触发 memory update

并不是每个位置都更新 memory。实现里只有满足以下条件的位置才会产生有效写入：

1. token 位置已经超出窗口，即 $t \ge W$
2. 若设置了 `mem_update_stride = S`，则只在满足

$$
(t-W) \bmod S = 0
$$

的位置更新

因此，代码里实际构造的是 `gates` 和 `updates` 两个张量：

$$
\mathbf{G} \in \mathbb{R}^{B \times T \times H_{\mathrm{kv}} \times M},
\qquad
\mathbf{U} \in \mathbb{R}^{B \times T \times H_{\mathrm{kv}} \times M \times d_h}.
$$

对不更新的位置，默认取：

$$
\mathbf{g}_t = \mathbf{1},
\qquad
\mathbf{u}_t = \mathbf{0},
$$

此时递推自动退化为

$$
\mathbf{m}_t = \mathbf{m}_{t-1},
$$

因此整个 memory scan 可以统一在所有 token 上执行，而不需要显式跳过无效位置。

---

### 5. Memory Read：如何读取记忆

memory state 用于构造额外的 memory keys / values。先做可选归一化：

$$
\tilde{\mathbf{m}}^{(h,s)}_t
=
\mathrm{Normalize}\!\left(\mathbf{m}^{(h,s)}_t\right),
$$

再定义：

$$
\mathbf{k}^{(h,s)}_{\mathrm{mem},t} = \tilde{\mathbf{m}}^{(h,s)}_t,
\qquad
\mathbf{v}^{(h,s)}_{\mathrm{mem},t} = \alpha \tilde{\mathbf{m}}^{(h,s)}_t,
$$

其中 $\alpha = \exp(\log \mathrm{mem\_scale}) > 0$ 是可学习标量。

由于 query 头数可能大于 key-value 头数，memory keys / values 在实现里会按 `num_kv_groups` 被 broadcast 到 query heads。

---

### 6. Local 分支与 Memory 分支如何融合

GM-SWA 不是把 memory slot 直接拼到局部 window 后统一做 softmax，而是先分别得到 local 分支和 memory 分支的输出，再做 log-sum-exp 融合。

设：

- 局部分支输出为 $\mathbf{o}^{\mathrm{local}}_t$
- 局部分支 log-normalizer 为 $\ell^{\mathrm{local}}_t$
- memory 分支输出为 $\mathbf{o}^{\mathrm{mem}}_t$
- memory 分支 log-normalizer 为 $\ell^{\mathrm{mem}}_t$

则总的归一化项是

$$
\ell^{\mathrm{tot}}_t
=
\log\left(
\exp(\ell^{\mathrm{local}}_t)+
\exp(\ell^{\mathrm{mem}}_t)
\right).
$$

最终输出为

$$
\mathbf{o}_t
=
\exp(\ell^{\mathrm{local}}_t-\ell^{\mathrm{tot}}_t)\,\mathbf{o}^{\mathrm{local}}_t
+
\exp(\ell^{\mathrm{mem}}_t-\ell^{\mathrm{tot}}_t)\,\mathbf{o}^{\mathrm{mem}}_t.
$$

这和代码里的 `torch.logaddexp` 以及 `local_weight / mem_weight` 完全对应。

#### 单 slot 情况：$M=1$

当 $M=1$ 时，每个 key-value 头只有一个 memory vector，memory logits 为

$$
s^{\mathrm{mem}}_{t,h}
=
\frac{\langle \mathbf{q}_{t,h}, \mathbf{k}^{\mathrm{mem}}_{t,h}\rangle}{\sqrt{d_h}}.
$$

这时：

$$
\ell^{\mathrm{mem}}_t = s^{\mathrm{mem}}_t,\qquad
\mathbf{o}^{\mathrm{mem}}_t = \mathbf{v}^{\mathrm{mem}}_t.
$$

#### 多 slot 情况：$M>1$

当 $M>1$ 时，同一 key-value 头内先在 slot 维度做 softmax：

$$
\pi^{(h,s)}_t
=
\mathrm{softmax}_s
\left(
\frac{\langle \mathbf{q}_{t,h}, \mathbf{k}^{(h,s)}_{\mathrm{mem},t}\rangle}{\sqrt{d_h}}
\right),
$$

再得到

$$
\mathbf{o}^{\mathrm{mem}}_{t,h}
=
\sum_{s=1}^{M}
\pi^{(h,s)}_t \mathbf{v}^{(h,s)}_{\mathrm{mem},t},
$$

以及

$$
\ell^{\mathrm{mem}}_{t,h}
=
\log\sum_{s=1}^{M}
\exp\left(
\frac{\langle \mathbf{q}_{t,h}, \mathbf{k}^{(h,s)}_{\mathrm{mem},t}\rangle}{\sqrt{d_h}}
\right).
$$

---

### 7. 与纯 SWA baseline 的关系

当设置 `disable_memory = true` 时：

- 不生成 gate / memory update
- 不维护 recurrent memory state
- 只保留纯 sliding-window local 分支

因此当前实现里的 SWA baseline 与 GM-SWA 在以下方面保持一致：

- block 结构
- tokenizer
- optimizer / scheduler
- 窗口大小 $W$
- hidden size / 层数 / 参数规模

唯一差异就是 memory path 是否存在。这一点对做 ablation 很重要。

---

## 二、训练上的算子加速

### 1. 为什么 memory update 不能直接用 Python 递推

上面的递推

$$
\mathbf{m}_t = \mathbf{g}_t \odot \mathbf{m}_{t-1} + (1-\mathbf{g}_t)\odot \mathbf{u}_t
$$

在推理时逐 token 执行没有问题，因为每次只更新一个状态。但训练时需要整段序列上所有时刻的 memory states：

$$
(\mathbf{m}_1,\dots,\mathbf{m}_T).
$$

如果直接用 Python for-loop 在训练图里展开：

```python
for t in range(T):
    m = g_t * m + (1 - g_t) * u_t
```

会有两个问题：

1. 序列维完全串行，GPU 并行度很差
2. `varlen` 打包训练下，长序列会明显拖慢 step time

因此 memory 分支必须有一个并行 scan 版本。

---

### 2. 旧的闭式 scan 形式

把递推写成标量化形式，定义

$$
\mathbf{p}_t = \prod_{i=1}^{t}\mathbf{g}_i.
$$

则可以推出

$$
\mathbf{m}_t
=
\mathbf{p}_t \odot
\left(
\mathbf{m}_0 +
\sum_{i=1}^{t}
\frac{(1-\mathbf{g}_i)\odot \mathbf{u}_i}{\mathbf{p}_i}
\right).
$$

实现上对应：

1. 对 $\log \mathbf{g}_t$ 做前缀和

$$
\log \mathbf{p}_t = \sum_{i=1}^{t}\log \mathbf{g}_i
$$

2. 再指数化得到 $\mathbf{p}_t$
3. 构造

$$
\mathbf{c}_t
=
\frac{(1-\mathbf{g}_t)\odot \mathbf{u}_t}{\mathbf{p}_t}
$$

4. 对 $\mathbf{c}_t$ 再做一次前缀和

$$
\mathbf{a}_t = \sum_{i=1}^{t}\mathbf{c}_i
$$

5. 最终恢复

$$
\mathbf{m}_t = \mathbf{p}_t \odot (\mathbf{m}_0 + \mathbf{a}_t)
$$

这正是当前代码在 **无梯度路径** 下仍然采用的 fused 形式，对应：

- `chunk_global_cumsum_scalar`
- `chunk_global_cumsum_vector`

这条路径的优点是很快；缺点是如果用于训练，$\mathbf{p}_t$ 很小的时候会出现非常大的中间量 $\mathbf{c}_t$，数值稳定性差，长序列下容易把梯度范数冲坏。

---

### 3. 当前训练使用的稳定 affine prefix-scan

为了解决上面的数值问题，同时保住并行性，当前训练实现把递推改写成仿射变换：

$$
\mathbf{m}_t = \mathbf{A}_t \odot \mathbf{m}_{t-1} + \mathbf{b}_t,
$$

其中

$$
\mathbf{A}_t = \mathbf{g}_t,\qquad
\mathbf{b}_t = (1-\mathbf{g}_t)\odot \mathbf{u}_t.
$$

现在把每个时刻看成一个仿射对 $(\mathbf{A}_t,\mathbf{b}_t)$。两个连续更新的复合仍然是仿射的：

$$
(\mathbf{A}_2,\mathbf{b}_2)\circ(\mathbf{A}_1,\mathbf{b}_1)
=
\left(
\mathbf{A}_2\odot \mathbf{A}_1,\;
\mathbf{A}_2\odot \mathbf{b}_1 + \mathbf{b}_2
\right).
$$

这个复合是结合的，因此可以做 prefix-scan。

如果定义长度为 $t$ 的前缀复合结果为 $(\bar{\mathbf{A}}_t,\bar{\mathbf{b}}_t)$，那么

$$
\mathbf{m}_t = \bar{\mathbf{A}}_t \odot \mathbf{m}_0 + \bar{\mathbf{b}}_t.
$$

这个公式与原始递推完全等价，但避免了

$$
\frac{1}{\prod_i \mathbf{g}_i}
$$

这种不稳定中间量。

---

### 4. 当前实现里的并行扫描方式

当前训练代码使用的是 **Hillis-Steele 风格的并行前缀扫描**。设 token 维长度为 $T$，初始时：

$$
\mathbf{A}^{(0)}_t = \mathbf{g}_t,\qquad
\mathbf{B}^{(0)}_t = (1-\mathbf{g}_t)\odot \mathbf{u}_t.
$$

然后对于 offset

$$
1, 2, 4, 8, \dots
$$

依次做：

$$
\mathbf{A}^{(\ell+1)}_t =
\mathbf{A}^{(\ell)}_t \odot \mathbf{A}^{(\ell)}_{t-\mathrm{offset}},
$$

$$
\mathbf{B}^{(\ell+1)}_t =
\mathbf{A}^{(\ell)}_t \odot \mathbf{B}^{(\ell)}_{t-\mathrm{offset}}
+
\mathbf{B}^{(\ell)}_t,
\qquad t \ge \mathrm{offset}.
$$

最终得到整段前缀复合：

$$
\bar{\mathbf{A}}_t,\bar{\mathbf{B}}_t.
$$

再恢复：

$$
\mathbf{m}_t = \bar{\mathbf{A}}_t \odot \mathbf{m}_0 + \bar{\mathbf{B}}_t.
$$

这正对应代码里的：

- `prefix_gates`
- `prefix_biases`
- `offset <<= 1`

虽然这个实现的总算术复杂度是 $O(T \log T)$，但它消除了 Python 级串行递推，实际在 GPU 上远快于 naive recurrent loop，而且稳定性明显更好。

---

### 5. 变长 `varlen` 训练如何处理

对于 packed sequences，不能让一个样本的 memory 传到另一个样本里。当前实现通过 `cu_seqlens` 构造每个 token 的序列编号：

$$
\mathrm{seq\_id}(t) \in \{0,\dots,N-1\}.
$$

当扫描 offset 为 $k$ 时，只允许同一序列内部做复合：

$$
\mathbb{1}_{t,k}
=
\mathbf{1}\!\left[\mathrm{seq\_id}(t)=\mathrm{seq\_id}(t-k)\right].
$$

于是变长版更新变成

$$
\mathbf{A}'_t =
\begin{cases}
\mathbf{A}_t \odot \mathbf{A}_{t-k}, & \mathbb{1}_{t,k}=1, \\
\mathbf{A}_t, & \mathbb{1}_{t,k}=0,
\end{cases}
$$

$$
\mathbf{B}'_t =
\begin{cases}
\mathbf{A}_t \odot \mathbf{B}_{t-k} + \mathbf{B}_t, & \mathbb{1}_{t,k}=1, \\
\mathbf{B}_t, & \mathbb{1}_{t,k}=0.
\end{cases}
$$

因此 packed batch 里的每个子序列都在自己的边界内独立扫描。

这和代码中的

```python
valid = seq_ids[offset:] == seq_ids[:-offset]
```

完全一致。

---

### 6. 当前实现的“训练路径”和“无梯度路径”

当前 memory scan 有两条实现路径：

#### 6.1 训练路径

当 `gates.requires_grad` 或 `updates.requires_grad` 为真时，走：

$$
\texttt{\_run\_memory\_scan\_torch}
$$

即上面介绍的稳定 affine prefix-scan。

特点：

- 可微
- 数值稳定
- 支持 `varlen`
- 与 token-by-token recurrence 等价

#### 6.2 无梯度路径

当不需要梯度时，走：

$$
\texttt{\_run\_fused\_memory\_scan}
$$

里的 fused cumsum 路径，即前面那套闭式 scan：

$$
\log \mathbf{p}_t \rightarrow \mathbf{p}_t \rightarrow \mathbf{c}_t \rightarrow \mathbf{a}_t \rightarrow \mathbf{m}_t.
$$

特点：

- 更快
- 适合无梯度阶段
- 不适合直接作为训练反向路径

这也是为什么当前实现同时保留两套 scan：**训练时用稳定仿射 scan，推理/无梯度时用 fused cumsum。**

---

---

### 8. 这一版实现解决了什么问题

这轮实现最终解决的是三个问题：

1. **新增 memory 参数真正参与训练**
   - `gate_net`
   - `mem_proj`

2. **训练路径不再依赖不稳定的闭式除法中间量**
   - 避免长序列下梯度范数异常

3. **速度不再退回到 naive recurrent loop**
   - 用并行 prefix-scan 保住训练吞吐

因此，从论文写作角度，可以把当前版本概括为：

> GM-SWA 的核心不仅是“在 sliding window 上加一个 gated memory”，还包括“为该 memory recurrence 提供一个数值稳定、支持 packed varlen、可并行训练的 prefix-scan 实现”。

---

## 三、可以直接转成论文的方法点

如果后面要把这份说明并回论文，最值得保留的点是：

1. **算法定义**
   - local SWA 不变
   - memory 从 evicted values 写入
   - gate 控制 retain / overwrite
   - local 与 memory 用 log-sum-exp 融合

2. **训练算子**
   - 原递推是 affine recurrence
   - affine pair 的复合满足结合律
   - 因此可做 prefix-scan
   - packed varlen 用 sequence-boundary mask

3. **工程价值**
   - 既修复“参数没梯度”的问题
   - 又避免数值不稳定
   - 同时保持训练速度
