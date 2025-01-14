Code adapted from the paper "Low-Resource" Text Classification: A Parameter-Free Classification Method with Compressors
Paper: https://aclanthology.org/2023.findings-acl.426/.

Changes made to the original codebase:
- Introduction of the random seed as a parameter to the script, to make things reproducible
- Prints were changed such that the output corresponds to a valid csv file
- Creation of a file "experiment.sh" that fully defines the experiment to conduct, again for reproducibility
- Additional option of using lz4 as compression algorithm
- A new algorithm that does classification based on how many of the rare words are similiar between the texts (more detail below), to be used when calling main_text.py with `--compression hashed_ngram`.

The shell script is to be called with 
```
./experiment.sh > results.csv
```
in order to directly save the results to a csv file. Note that if some error occurs while running the script (or if unexpected prints are encountered), the csv file will no longer be valid.

I wanted to make as little changes to the original codebase as possible, hence the hacky csv-stuff.

### What's the point?
The original paper contains a contradiction: On one hand, they argue that their use of the Kolmogorov complexity as a measure is why it works. But this would imply that a better compression ratio (and thus a better approximation of the Kolmogorov complexity) would lead to better performance. Their experimental results contradict this, as the method with the best compression ratio, bz, performs worse than gzip. Their explanation, that bz dismisses character order, seems plausible but also suggests that there is something else than only compression that makes their method work.

`gzip` is actually a combination of something called `lz77` and huffman coding. I wanted to see whether the `lz77`-part is already enough to make their method work. Since there is no fast implementation of `lz77` available, I resorted to lz4 instead. Since their flags `--all_train` and `--all_test` do not work on my machine, I limit myself to comparing the methods in what they call their "few-shot setting" with n-shots = 100. And in fact we do see that `lz4` achieves performance very similar to `gzip` - sometimes it's better, sometimes worse (note also that I can reproduce the results in their Figure 3, as a sanity check).

This finding further contradicts their initial hypothesis that a good estimate of the Kolmogorov complexity is the reason of why their method works, as `lz4` has a compressio ratio which is much worse than `gzip`. An alternative hypothesis seems much more plausible: that their method is just a way of comparing how many character-level n-grams co-occur in two distinct documents. 

### An algorithm to compare character-level n-grams
To verify this, we implement the following algorithm: 
1. For all texts, extract all character n-grams within the texts with 5 ≤ n ≤ 50.
2. For the extracted n-grams, calculate an 8-digit hash code using `hash(n_gram) % int(10e8)` in python.
3. Store these hashed values as a set for every data set entry.

Training: For all texts in the training set belonging to the same topic, create a set of hash codes that are relevant to said topic by using the set union on all the entries. This can be precomputed once.

Inferencing: At inference time, we take some text, calculate the set of its hashed n-grams, and calculate the size of its intersection with the "learned" topic-specific sets. We assign the query text to the topic with the largest intersection.

Notes: In addition to this algorithm being more honest about what it's actually doing, its inference time is constant w.r.t. the size of the training set, whereas the costs of the algorithm proposed in the original paper grows linearly with the training set - making it not useful in practice. 

### The results
|dataset|algorithm|mean accuracy|stdev|
|-|-|-|-|		
|AG_NEWS|gzip|0.737300|0.004907|
|AG_NEWS|hashed_ngrams|0.745850|0.008512|
|AG_NEWS|lz4|0.779150|0.005923|
|DBpedia|gzip|0.859957|0.001847|
|DBpedia|hashed_ngrams|0.902500|0.004846|
|DBpedia|lz4|0.887271|0.003049|
|YahooAnswers|gzip|0.353660|0.005507|
|YahooAnswers|hashed_ngrams|0.393460|0.018398|
|YahooAnswers|lz4|0.359760|0.003715|

### What does this _mean_?
But why does this work so well, and why didn't we have this idea earlier? My guess is that this is a good way of looking at the long tails (i.e. the very rare words and character-combinations) of the character distributions. Meanwhile, TF-IDF-like methods will neglect these tails due to the low "term frequency", and attention-based neural networks seem also to struggle quite a bit to learn long-tail information (as demonstrated in this paper: https://arxiv.org/pdf/2211.08411.pdf for QA tasks).
