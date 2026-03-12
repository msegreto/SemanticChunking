# Chunking Evaluation Framework
This is an overview of what the framework should evaluate and consider. Starting from the framework in "Is Semantic Chunking Worth the Computational Cost?" and adding experiments and components that emerged during the discussion. I divided the framework by components that are sequentially linked.

**Superscripts in section titles refer to the references used to define each module. Numbers in square brackets [n] indicate that the methodology follows the approach described in the corresponding reference.**

## 1. Storage dataset (experimental variable 1)
Simply we have several datasets stored that we can test running the framework evaluation.
(Dataset research)

## 2. Document split $^1$ $^,$ $^2$ (experimental variable 2)
Here we have to decide if split at sentence level[1] or follow[2] for the proposition level split.
If we exploit the second option we should avoid the next phases( w.r.t 3,4,5)

## 3. Router  $^3$ $^,$ $^4$ (experimental variable 3)
This module exploits the use of a transformer(?,[4?]) in order to decide the dimension that the chunk should have.
Q1: Is there a way that allows us to discover which is the optimal length of the chunk a priori?

## Chunking Strategies  $^1$ $^,$ $^3$ $^,$ $^?$(experimental variable 4)
In this module we chunk the document. At the moment the possible chunking strategies are:
1. fixed[1]
2. semantic breakpoint[1]
3. clustering [1]
4. meta chunker [3] (*works good but is heavy*)
5. late(?)[?]

Q2: is there a chunking strategy which outperform on web pages?

## 5. Intrinsic Evaluation  $^3$ 
Here we evaluate and store the metrics of [3] the produced chunks.
1. Chunk Stickness
2. Boundary Clarity


*Idea: could it be an option stop the experiment if these metrics are under a certain threshold?* 
*could it lead to a time and resource optimization?*

## 6.Embedding $^1$ $^,$ $^?$ (experimental variable 5, experimental variable 6)
Here there are some things to take into account:
1. Search for the optimistic embedder => different embedding, different context length =?=> different results. Should we use the ones in [1] ?
2. Q3: Is it possible to introduce a second router can based on the document that classify the optimistic pooling strategies?


## 7. Extrinsic Evaluation on Proxy metrics  $^1$ 
Follow the paper evaluation metrics [1].
1. document retrieval
2. evidence retrieval
3. answer generation
4. information based performance evaluation from[2?]

*Doubt: during the reunion it has been decided to use NDCG but in [1] they explicitly told to use F1 because in this situation better addresses what we want to evaluate?*

## Research question 
Q1: Is there a way that allow us to discover which is the optimal lenght of the chunk apriori?
Q2: On web pages is there a chunking strategy which outperform?
Q3: Is it possible to introduce a second router can based on the document that classify the optimistic pooling strategies?

# REFERENCES
1. Is Semantic Chunking Worth the Computational Cost? (given)
2. Dense X Retrieval: What Retrieval Granularity Should We Use? (given)
3. MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System (from today discussion)
4. Transformer over Pre-trained Transformer for Neural Text Segmentation with Enhanced Topic Coherence (from today discussion)
5. ??? (given but i didn't get the title)
