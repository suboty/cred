# Regular Expressions Clustering

**Cluster analysis** or **clustering** is the task of grouping a set of objects 
in such a way that objects in the same cluster 
are more similar to each other according to some similarity function than to those in other clusters.

To run clustering do this:
```shell
python3 clustering.py -v -u --algname encoder_name --filter filter_word_for_input_data
```
- -u for update visualizations
- -v for verbose examples printing
- --algname: "bert" or "tf_idf"
- --filter for filtering input data (looking for in description on title regexes)

## Clustering results
In *clustering_reports*