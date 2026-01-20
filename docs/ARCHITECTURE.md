# Architecture, v.0.0.1

## Pipline 

### Gathering data diagram 

```mermaid
graph TD
A[Call for model to make embedding] --> B[Put embedding to the vector DB]
```

### Get the persona notes from the diagram (simplified)

```mermaid
graph TD
B[transform the user request to embedding] --> C[get the data from the vector DB]
```

## Cloude Models to make embeddings

### OpenAI -- Embeddings API

### Cohere - Embed API

### Hugging Face Interference API (Sentence-Transformers)


## Vector DBs - self hosted

### Qdrant(OSS)

### Chroma 


## Intergation tool to handle 

## LangChain