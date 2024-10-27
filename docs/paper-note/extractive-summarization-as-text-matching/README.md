# Paper Note -- Extractive Summarization as Text Matching
## Introduced Background Knowledges

## Basic Ideas
### Sentence Level VS Summary Level
The quality of summary will be quantified by some metrics, in paper author 
used **ROUGE**. Sentence level extraction extract "highest quality sentences" 
based on some method and merge them together as final summary. Each sentence 
maybe "locally optimized", by the combination of them maybe not "globally 
optimized" (like greedy decoding VS beam search). For **summary level scoring** 
method, they will rank several *condidate summaries* and the top summary 
will be closer to "globally optimized" summary.
  
Author quantified the performance gap between **sentence levelex traction** 
and potentional best summary by including 2 new concepts **pearl summary** 
and **best summary**,  also 2 new definitions **sentence level score** and 
**summary level score**. Plotting a bar plot, x-axis is the ratio of the 
samples which **best summary** can be extracted by sentence level extraction 
method, and we can observe there's a significant propertion that sentence 
level extraction results are not the most optimized.

## Siamese-BERT
**Siamese-BERT** is used to judge how good the **candidate summary** is with 
a giver source document. It's a double-tower architecture, there two towers 
sharing a single BERT like encoder. It's modeling 3 factors at same time:
* **Relevance between source document and candidate summary**: 
    * `f(source_doc, candidate_summary) = cosine(source_doc; candidate_summary)`
* **Relative quality between golden summary and candidate summary**: 
    * The ideas is **golden summary** should be semantically closest to the 
      source document, which means:
        * When candidate is closer to source document compare with 
          golden summary, we can stop optimize on that sample to avoid 
          over-fitting risk and force learning focus on other samples.
    * **margin-based triplet loss** 
      `L1=max(0; f(source_doc; candidate_summary) - f(source_doc; golden_summary) + r1);`
* **Modeling candidate summaries' ranking information with **learning to rank**:
    * Candidate summaries ranking is based on **ROUGE**.
    * High ranking candidates should has a higher cosine similarity.
    * The larger ranking gap candidate pair should have larger cosine similarity 
      gap and **margin**.
    * **Pairwise margin loss**: 
      `L2 = max(0, f(source_doc; lower_rank_candidate_summary) - f(source_doc; higher_rank_candidate_summary) - ranking_gap * r2)`

## Solved Problems
TBD

## Questions to Follow Up
* How to generate multiple candidate summaries (with extractive method)?


## Appendix
### Keywords/Concepts
* Neural Extractive Summarization
    * Sentence Level Extraction
    * Summary Level Scoring
* Semantic Matching Problem
* Pairwise Margin Loss
