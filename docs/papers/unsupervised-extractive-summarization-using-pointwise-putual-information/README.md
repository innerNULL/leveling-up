# Paper Note -- Unsupervised Extractive Summarization using Pointwise Mutual Information
## Relevant Works (Pre-Reading)
* [Simple Unsupervised Keyphrase Extraction using Sentence Embeddings](https://github.com/innerNULL/notes/tree/main/docs/papers/simple-unsupervised-keyphrase-extraction-using-sentence-smbeddings)

## Overview
Similar with [Simple Unsupervised Keyphrase Extraction using Sentence Embeddings](https://github.com/innerNULL/notes/tree/main/docs/papers/simple-unsupervised-keyphrase-extraction-using-sentence-smbeddings), this work also decouple the criteria of keyphrases selection into to dimension: **relevance(informativeness)** and **redundancy**. 


## Innovations
### Not Using Semantic Similarity
Comprare with previous work [Simple Unsupervised Keyphrase Extraction using Sentence Embeddings](https://github.com/innerNULL/notes/tree/main/docs/papers/simple-unsupervised-keyphrase-extraction-using-sentence-smbeddings), instead of using **semantic similarity**, this work proposed a noval approach, which uses **pointwise mutual information(PMI)** to quantify **relevance** and **redundancy**.

**Note**:
* Semantive similarity based methods are **similarity based**, so need **decoder only** model to get vector representation of document and phrases.
* The proposed **PMI** based methods is **probability based**, so instead of decoder-only model, it relies on generative LM (**decoder-only or encoder-decoder models**).

### Migrate PMI Defination to Text Data with LM (Language Model)
#### Notations
* General **PMI** on Condition $x$ and $y$,  $pmi(x; y)$:
    * $pmi(x; y) = log_{2} \dfrac{p(x, y)}{p(x) \cdot p(y)} = log_{2} \dfrac{p(y|x)}{p(y)} = log_{2} \dfrac{p(x|y)}{p(x)}$
    * Based on above point, $pmi(x; y) = pmi(y; x)$
* **Sequence (Text/Sentence) Probability** $p_{LM}(s)$: **LM (Langeuage Model)** models the probability of a token sequence, so with LM we can get $p_{LM}(s)$.

#### Explanations
* How to understand $pmi(x; y) = log_{2} \dfrac{p(y|x)}{p(y)}$?
    * [Defination](https://en.wikipedia.org/wiki/Pointwise_mutual_information), **mutual information** can be known as overlap between information contained by $x$ and $y$.
    * The difference (or ratio) of the probability of $y$ condition on $x$ and probability of $y$ alone.
        * If $p(y|x)$ (much) larger than $p(y)$: 
            * $y$ is more likely to happen after $x$ happened priorly (condition on $x$) than before.
            * Above point means $x$ must contains some information which can infer or trigger $y$, so $x$ and $y$ has higher mutual information.
        * If $p(y|x)$ (much) smaller than $p(y)$: Similar with above analysis, this case implies the probability of $y$ is smaller afer $x$ happened before, so with $x$ it's hard to infer $y$, so smaller mutual information.
* How to migrate PMI calculation on text data?
    * The key part of PMI is how to calculate **conditional probability** (for text data) $p_{LM}(text_i|text_j)$.
    * For language model (LM), it's easy to understand $p_{LM}(text_i|text_j)$ as the probability of $text_i$ when using $text_j$ as prefix, so:
        * $p_{LM}(text_i|text_j) = \dfrac{p(text_i, text_j)}{p(text_j)} = \dfrac{p(concat(text_j, text_i))}{p(text_j)}$
    * $p(text)$ can be calculated by any generative LM (**docoder-only, encoder-decoder** model).
* Why can use PMI as metric for keyphrase selection? 
    * It's natural to view $text_i$ as redundency of $text_j$ when we has higher probability to infer $text_i$ from $text_j$.


## Algorithm
### Steps
* **Segmentation**: Split source text $doc$ into sentences/phrases.
* **Initialize Variables**
    * $S$: Selected keyphrases, initialized as an empty set.
    * $k$: Final target is to select top $k$ keyphrases at most.
    * $\lambda_1$: Weight of **relevance (informativeness) score**.
    * $\lambda_2$: Weight of **redundancy score**.
* **Greedy Sequential Selection**, while $S$ contains less than $k$ phrases:
    * For each phrase $s$ in splitted phrase, calculate it's **importancy score**:
        * **Relevance(Informativeness) Score**: $pmi_{{relevance}}(s_i; d) = log \dfrac{p_{LM}(d|s)}{p_{LM}(d)}$
        * **Redundancy Score**: $pmi_{{redundancy}}(s_i; s_j) = log \dfrac{p_{LM}(s_j|s_i)}{p_{LM}(s_j)}, if\space i < j$, where $i, j$ are phrase indexs in $s$. According PMI defination, we just set $pmi_{{redundancy}}(s_i; s_j) = pmi_{{redundancy}}(s_j; s_i), if\space i > j$. 
        * **Importancy Score**: $\Delta(s_i) = \lambda_1 \cdot pmi_{{relevance}}(s_i; d) + \lambda_2 \cdot \sum_{s'  \in S} pmi_{{redundancy}}(s'; s_i)$
    * Select Most Important Phrase: Insert $\arg \max_{s_i \in doc}  \Delta(s_i)$ into $S$.
* $S$ is the final results, which contains $k$ extracted most important phrases.


## Implementations
See [mi-unsup-summ](https://github.com/vishakhpk/mi-unsup-summ)