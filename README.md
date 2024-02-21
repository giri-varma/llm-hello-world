# llm-hello-world

## Pre-requisites
1. Install python3.9 and pip before setting up the project-specific dependencies.
2. Install [langchain](https://python.langchain.com/docs/get_started/installation) LLM framework.
    ```shell
    /usr/bin/pip3 install --user gpt4all langchain 
    ```

## Basic
Simple AI chat bot.
```shell
/usr/bin/python3 basic.py
```

## Contextual
Contextual Q&A chat bot based on some sample PDF data like K8s documentation.

### Build
Build the corpus and the vector index. Sample data is hard-coded in `data/k8s-docs`. Can be updated for any directory with PDF data.
```shell
/usr/bin/python3 build.py
```

### Run
Build the corpus and index before running the query. The query is hard-coded. Can be updated to prompt the user for the query at runtime.
```shell
/usr/bin/python3 query.py
```