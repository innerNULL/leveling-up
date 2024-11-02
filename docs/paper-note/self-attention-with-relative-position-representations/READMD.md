# Paper Note - [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)

## Overview
Previously, absolute position information representations are used to encode the position information of tokens in transformer. 

One typical **absolute position encoding** is to use **sinusoid function** to encode the position information of tokens, which is defined as follows:
$$
PE_{(pos,2i)}=sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$
$$
PE_{(pos,2i+1)}=cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$
where $pos$ is the position of token, $d_{model}$ is the dimension of token embedding.

Besides there're also some works using **learned absolute position encoding**, which is trainable during training process.

**Both above approachs are not generalizable to variable-length input (e.g., when in inferences time the tokens length islonger than the training time)**. So in this work the authors proposed to use **relative position representations** to encode the position information of tokens in transformer to eliminate above drawback. 

## Summary
### Integrate Position Encoding Into Self-Attention
Following are the formulas of self-attention machanism without position information injected:
$$
\begin{align}
z_{i} &= \sum_{i=1}^{n}\alpha_{i, j}(x_{j}W^{v}) \\
\alpha_{i, j} &= \frac{\textbf{exp}(e_{i, j})}{\sum_{k=1}^{n}\textbf{exp}(e_{i, k})} \\ 
e_{i, j} &= \frac{(x_{i}W^{Q})(x_{j}W^{K})^T}{\sqrt{d_{z}}}
\end{align}
$$
where $e_{i, j}$ is the attention score, $x_{i}$ is the input token, $W^{Q}$, $W^{K}$, $W^{V}$ are the linear transformation matrix, $d_{z}$ is the dimension of the output token.

First, here we consider the scenario only integrate position encoding to the **value** and **key**, the position information injection methods can refer to following formulas:
$$
\begin{align}
z_{i} &= \sum_{i=1}^{n}\alpha_{i, j}(x_{j}W^{v} + a_{i, j}^V) \\
e_{i, j} &= \frac{(x_{i}W^{Q})(x_{j}W^{K} + a_{i, j}^K)^T}{\sqrt{d_{z}}}
\end{align}
$$
Here $a_{i, j}^V$ and $a_{i, j}^K$ are vectors represents relative position information. $a_{i, j}^*$ is a vector sourced from one of a relative position matrix $w_{k}^V$ or $w_{k}^K$. Here $k$ (lower case) is the relative position offset between token $i$ and $j$. So the $a_{i, j}^*$ can be computed as: 
$$
\begin{align}
a_{i, j}^* &= w_{i-j}^*    
\end{align}
$$

### Clipping
As $[w_{i, j}^*]^{d_{a} \times {\textbf{offset}}_{\textbf{max}}}$ is a matrix, it has fixed dimentions, so when $i$ and $j$ are too far, their relative offset may out of ${\textbf{offset}}_{\textbf{max}}$, so here we update equation (6) to (7) by integrating a "clipping" operation:
$$
\begin{align}
a_{i, j}^* &= w_{\textbf{clip}(i-j, {\textbf{offset}}_{\textbf{max}})}^* \\
clip(x, k) &= max(-k, min(k, x)) 
\end{align}
$$ 

The suggestion here is, even though theoretically each token can capture the different between all other tokens' relative positions, but according to the original paper:
> "We hypothesized that precise relative
position information is not useful beyond a
certain distance. Clipping the maximum distance
also enables the model to generalize to sequence
lengths not seen during training" 

Therefore, according (7) and (8), in this work we consider $2k + 1$ unique relative positions.

### How to Understand and Leverage $[w_{i, j}^*]^{d_{a} \times {\textbf{offset}}_{\textbf{max}}}$
Now another question is, how to uderstand $i - j$ especially when $i - j < 0$? This will happen when token $i$ is before token $j$.

As we discussed before, we have two matrixs of position encodings for key and value matrixs in self-attention mechanism:
$$
\begin{align}
[w_{i, j}]^K &= (w_{-k}^K, w_{-k + 1}^K, \cdots, w_{k}^K) \\
[w_{i, j}]^V &= (w_{-k}^V, w_{-k + 1}^V, \cdots, w_{k}^V)
\end{align}
$$
where $k >= 0$.

Here maybe looks confusion as sometimes we have $k < 0$, but as $[w_{i, j}^*]$ have fixed dimensions, we can always mapping each $k$ to a meaningful matrix index.


### Relative Position Embedding Sharing
In paper they listed some time complexities, but simply speaking, to improve the computationl efficiency, we can **share the relative position embedding matrixs in different attention heads**.

## Others
### Meaningless Integration the Concept
The authors tried to using graph to explain the concept of relative position encoding, but it's not necessary, so can just forgot it.

