# Regular Expressions Clustering

**Cluster analysis** or **clustering** is the task of grouping a set of objects 
in such a way that objects in the same cluster 
are more similar to each other according to some similarity function than to those in other clusters.

## Encoders

First-of-all, choose a encoder for regexes (for vectorize representation).

### TF-IDF as encoder

TF (Term Frequency) measures how often a certain word appears in a given document. Thus, TF measures the importance of a word in the context of a single document.
IDF (Inverse Document Frequency) measures how unique a word is across a collection of documents. Words that appear in most documents have a low IDF because they do not contribute much information value.
The TF-IDF formula combines the concepts of TF and IDF to calculate the importance of each word in each document.

Three methods of obtaining TF-IDF vectors are proposed:
- Tokenization by regular expression symbols
- Tokenization by regular expression non-terminals
- Tokenization by custom tokens (lexical analyzer as in compilers)

#### PCA visualization
<p float="left">
  <img src="assets/tf_idf_chars_pca.png" width="300" />
  <img src="assets/tf_idf_non_terminals_pca.png" width="300" />
  <img src="assets/tf_idf_tokens_pca.png" width="300" />
</p>

#### UMAP visualization
<p float="left">
  <img src="assets/tf_idf_chars_umap.png" width="300" />
  <img src="assets/tf_idf_non_terminals_umap.png" width="300" />
  <img src="assets/tf_idf_tokens_umap.png" width="300" />
</p>
