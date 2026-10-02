# 多值 Delta 与低条件熵：局部监督和终局 Value 的信息量关系

Last updated: 10/02/2026

## 1. 研究目标与当前结论

我们希望研究：当每个 chunk 对应一个几乎可以由该 chunk 确定的多值 delta，而终局奖励需要结合全部 delta 时，直接学习终局 value 是否比学习局部 delta 更困难。

本笔记首先建立一个中间结论：**局部 delta 向量可以保留终局奖励中没有保留的信息，即使各 delta 存在少量条件不确定性。**

这个信息量结论尚不等于学习难度结论。本文只推导当前设定能够支持的关系，并指出还需要哪些额外条件。

## 2. 新的设定

### 2.1 Chunk 序列与稀疏依赖

记随机 chunk 序列为

$$
C=C_{1:n}=(C_1,\ldots,C_n).
$$

允许 chunk 之间存在稀疏依赖：

$$
P(C_{1:n})
=\prod_{t=1}^{n}P(C_t\mid C_{\operatorname{pa}(t)}),
$$

其中

$$
\operatorname{pa}(t)\subseteq\{1,\ldots,t-1\},
\qquad |\operatorname{pa}(t)|\le k\ll n.
$$

稀疏依赖不代表 chunk 相互独立，也不代表统计相关性一定很弱。下面的信息量推导不需要用到稀疏性；保留该假设，是为了描述所关注的问题结构。

### 2.2 多值、低条件熵的 Delta

令

$$
\Delta_t\in\mathcal D_t,
\qquad D=\Delta_{1:n}.
$$

$\mathcal D_t$ 可以包含多个数值，不再限定为二元集合。为使用离散熵，暂时假设这些集合有限，所有对数以 2 为底。

用以下条件替换严格的确定性假设：

$$
\boxed{H(\Delta_t\mid C_t)<\epsilon,\qquad \epsilon>0.}
$$

其含义是：给定对应 chunk 后，delta 的平均剩余不确定性很小。该条件允许 $H(\Delta_t\mid C_t)=0$；如果需要排除完全确定的情形，可以另外要求它严格大于零。

需要注意：

- 这是对输入分布取平均后的条件熵约束，不保证每一个具体 chunk 上的不确定性都小。
- 这个条件本身不保证 delta 与未来 chunk 条件独立。若要表达“delta 的生成只依赖本 chunk”，还可以假设 $P(\Delta_t\mid C_{1:n})=P(\Delta_t\mid C_t)$，但后面的熵界不需要这个额外假设。
- 不假设不同 delta 在给定完整 chunk 序列后相互独立，因此不能直接将其联合条件分布写成各项的乘积。

对于单个 delta，有

$$
I(\Delta_t;C_t)
=H(\Delta_t)-H(\Delta_t\mid C_t)
>H(\Delta_t)-\epsilon.
$$

若给定此前的 chunk，则

$$
\begin{aligned}
I(\Delta_t;C_t\mid C_{<t})
&=H(\Delta_t\mid C_{<t})-H(\Delta_t\mid C_{\le t})\\
&>H(\Delta_t\mid C_{<t})-\epsilon.
\end{aligned}
$$

这里不能一般地把 $H(\Delta_t\mid C_{<t})$ 替换为 $H(\Delta_t)$。

### 2.3 终局奖励与充分性

继续保留完整序列确定终局奖励的假设：

$$
R=F(C),\qquad R\in\{0,1\}.
$$

同时，假设全部 delta 对预测终局奖励充分：

$$
R\perp C\mid D.
$$

这两个假设共同推出

$$
H(R\mid D)
=H(R\mid C,D)+I(R;C\mid D)=0.
$$

因此，存在一个聚合函数 $G$，使得在分布支持集上几乎处处有

$$
R=G(D).
$$

**这些假设对 delta 的随机性有约束：同一个完整 chunk 序列产生的不同 delta 向量，都必须对应同一个终局奖励。** 随机变化可以发生在同一奖励类别内部，但不能导致奖励类别改变。

如需可加的解释，可以额外规定 $R=g(\sum_t\Delta_t)$。下面的推导不依赖可加性。

## 3. 整个 Delta 向量的条件熵

由条件熵链式法则，

$$
H(D\mid C)
=\sum_{t=1}^{n}H(\Delta_t\mid\Delta_{<t},C_{1:n}).
$$

增加条件不会增加离散条件熵，因此

$$
H(\Delta_t\mid\Delta_{<t},C_{1:n})
\le H(\Delta_t\mid C_t).
$$

求和得到

$$
\boxed{H(D\mid C)<n\epsilon.}
$$

于是，整个 delta 向量与 chunk 序列之间的互信息满足

$$
\boxed{I(D;C)>H(D)-n\epsilon.}
$$

这个结论不需要 chunk 独立，也不需要 delta 条件独立。需要留意的是，每个 delta 的误差预算会累积为 $n\epsilon$；单个位置上的小条件熵未必意味着整个向量的条件熵也很小。

## 4. 承上启下的关键结论

严格确定性的原设定给出

$$
I(R;C)=H(R)\le H(D)=I(D;C).
$$

在新的设定下，$H(D)=I(D;C)$ 一般不再成立。保留充分性后，数据处理不等式给出

$$
\boxed{I(R;C)=H(R)\le I(D;C)\le H(D),}
$$

并且

$$
\boxed{H(D)-I(D;C)=H(D\mid C)<n\epsilon.}
$$

因此，新设定下的 delta 向量依然至少保留了与终局奖励一样多的序列信息，同时它与完全确定情形之间的熵差受到 $n\epsilon$ 控制。

### 4.1 信息差的精确表达

由于 $R$ 是 $D$ 的函数，

$$
I(C;D,R)=I(C;D).
$$

再次使用互信息链式法则，

$$
\boxed{
I(D;C)-I(R;C)=I(D;C\mid R)\ge0.
}
$$

又因为 $R$ 也是 $C$ 的函数，

$$
\begin{aligned}
I(D;C\mid R)
&=H(D\mid R)-H(D\mid C,R)\\
&=H(D\mid R)-H(D\mid C).
\end{aligned}
$$

因此，

$$
\boxed{
I(D;C)-I(R;C)
=H(D\mid R)-H(D\mid C)
>H(D\mid R)-n\epsilon.
}
$$

这说明：delta 在同一终局奖励类别内部的变化，只有其中能被 chunk 解释的部分才构成额外信息。单纯增加与 chunk 无关的随机噪声，不能增加有用的互信息。

### 4.2 严格不等式的充分条件

若

$$
H(D\mid R)>n\epsilon,
$$

则必有

$$
\boxed{I(D;C)>I(R;C).}
$$

因为 $R$ 是 $D$ 的函数，$H(D\mid R)=H(D)-H(R)$，所以上面的条件也可以写成

$$
H(D)-H(R)>n\epsilon.
$$

这是一个充分条件，并非必要条件。严格不等式的精确条件是 $I(D;C\mid R)>0$。

如果不保留 delta 的充分性假设，仍然可以由 $R=F(C)$ 推出

$$
I(D;C)-I(R;C)>H(D)-H(R)-n\epsilon,
$$

但不再无条件保证 $I(D;C)\ge I(R;C)$。

## 5. 为什么暂时不能推出“Value 更难学”

低条件熵描述的是数据分布中的剩余不确定性。学习难度描述的是：不知道真实规律时，需要多少数据、何种模型或多少计算才能学到该规律。这是不同的问题。

一个反例可以说明当前假设不足以给出普遍的难度排序。令一个 chunk 为

$$
C=(X,Z),\qquad X\in\{0,1\},
$$

并定义

$$
\Delta=2f(Z)+X,\qquad R=X=\Delta\bmod2,
$$

其中 $Z$ 取有限个值，$f$ 是取多个非负整数值的未知函数。

这个构造满足：delta 是多值变量，$H(\Delta\mid C)=0<\epsilon$，完整 chunk 确定奖励，delta 也充分决定奖励。

但是，直接预测奖励只需要读取输入中的 $X$；预测 delta 还需要学习 $f$。当前假设并没有限制 $f$ 的复杂程度，因此不能排除 delta 预测任务比奖励预测任务更困难的情况。

所以，本节建立的信息量关系支持的是：**delta 标签可以携带额外的局部结构信息。** 它尚未证明这些额外信息更容易学，也未证明这些信息一定有助于学习所关注的 value。

## 6. 下一步需要明确的 Value 目标

“直接预测终局奖励的 value”可能指两个不同的目标。

### 完整序列上的 Value

$$
V_n(C)=\mathbb E[R\mid C]=F(C).
$$

此时没有终局奖励的条件随机性，但未知函数 $F$ 的学习复杂度仍然需要研究。

### 前缀上的 Value

$$
V_t(C_{\le t})
=\mathbb E[R\mid C_{\le t}]
=\mathbb E[G(D)\mid C_{\le t}].
$$

此时还要平均未观察贡献以及 delta 的剩余条件随机性。然而，对终局奖励存在不确定性，不等于其条件期望本身难以估计；例如恒定的条件期望可以很容易学习。

要进一步证明样本复杂度上的难度差异，需要先选定以上目标，并指定未知函数类、局部标签的可获得性、样本或标注预算以及误差指标。当前先不引入 Fano 不等式，也不声称已经完成这一学习难度证明。

## 7. 如果 Delta 是连续实数

以上推导使用离散 Shannon 熵。若 delta 是连续实数，不能直接将所有 $H$ 替换成微分熵：微分熵可能为负，确定性条件分布也可能是奇异的，前面的离散熵比较不能照搬。

有两种自然的建模方式：

1. 将 delta 按固定精度量化，然后对量化后的离散变量使用本文设定；相应的充分性需要重新检查。
2. 直接用均方误差刻画局部可预测性。若 delta 二阶矩有限，可以要求

$$
\mathbb E\!\left[
\left(\Delta_t-\mathbb E[\Delta_t\mid C_t]\right)^2
\right]<\epsilon.
$$

第二种方式更直接地约束局部预测误差，但需要另行建立整体 value 的误差比较，不能直接复用本文的熵差公式。

## 8. 当前可以引用的结论

假设局部 delta 为多值离散变量，满足 $H(\Delta_t\mid C_t)<\epsilon$，且完整 chunk 序列和完整 delta 向量均确定终局奖励，则

$$
I(R;C_{1:n})\le I(\Delta_{1:n};C_{1:n}),
$$

并且信息差满足

$$
I(\Delta_{1:n};C_{1:n})-I(R;C_{1:n})
>
H(\Delta_{1:n}\mid R)-n\epsilon.
$$

当同一奖励类别内部的 delta 熵大于累计条件熵预算时，delta 向量严格保留更多关于 chunk 序列的信息。这为研究局部监督提供了一个信息论上的中间结论；学习难度的比较仍需要额外的统计学习设定。
