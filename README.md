# Common Regular Expression Dataset

## Parse dataset

### all sources
```shell
sh create_cred.sh
```

### regex101
To start parsing <a href="https://regex101.com/library">this</a> page, you need to run the following command:
```shell
sh src/parse_regex101.sh
```

### regexlib
To start parsing <a href="https://www.regexlib.com">this</a> page, you need to run the following command:
```shell
sh src/parse_regexlib.sh
```
This script parse regex with SOAPAction <a href="https://www.regexlib.com/WebServices.asmx?op=ListAllAsXml">ListAllAsXml</a>.
Now run only with 500 samples.

## Use-Cases
### Regex Clustering
Example algorithms configuration:
```yaml
tf_idf:
  - tokens
  - chars
  - non_terminals
bert:
  - bert_base_uncased
  - codebert_base
  - modernbert_base
```

Code for clustering running:
```python
from pathlib import Path

from research.clustering import ClusteringUseCase

clustering = ClusteringUseCase(
    path_to_algorithms_yaml=Path(
        'path_to_algorithms_config'
    ),
    path_to_preprocessing=Path(
        'path_to_preprocessing_replacements'
    ),
    path_to_encoders=Path(
        'path_to_tokens_for_encoders'
    ),
    path_to_sql_queries=Path(
        'path_to_sql_queries_for_result_processing'
    )
)

input_regexes = [...]

result = clustering(input_regexes=input_regexes)
```

Clustering results will be saving in sqlite database **clustering_TIME_RUNNING.db**

Some statistics will be saved in **tmp** folder