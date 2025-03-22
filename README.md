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
