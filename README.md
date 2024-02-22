# llm-hello-world

## Pre-requisites
1. Install python3.9+ and pip before setting up the project-specific dependencies.
2. Install [langchain](https://python.langchain.com/docs/get_started/installation) LLM framework.
    ```shell
    /usr/bin/pip3 install --user gpt4all langchain langchain-openai beautifulsoup chromadb faiss-cpu langchainhub
    ```

## Chat 1
Simple AI chat bot using gpt4all directly.
```shell
/usr/bin/python3 chat1.py
```

## Chat 2
Simple AI chat bot using gpt4all via langchain framework.
```shell
/usr/bin/python3 chat2.py
```

## Chat 3
Simple AI chat bot using gpt4all server (which implements the same HTTP APIs as
OpenAPI) via OpenAPI chat via langchain framework. As a pre-req [run the gpt4all
API server in docker](https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-api#starting-the-app).
```shell
/usr/bin/python3 chat3.py
```

## Q&A 1
Contextual Q&A chat bot based on some sample PDF data like K8s documentation directly using GPT4All.

### Build
Build the corpus and the vector index. Sample data is hard-coded in `data/k8s-docs`.
Can be updated for any directory with PDF data.
```shell
/usr/bin/python3 qa1_build.py
```

### Query
Build the corpus and index before running the query. The query is hard-coded.
Can be updated to prompt the user for the query at runtime.
```shell
/usr/bin/python3 qa1_query.py
```

## Q&A 2
Contextual Q&A chat bot based on some web data using GPT4All server via langchain.
```shell
/usr/bin/python3 qa2.py
```