Intro

## Installation

    pip install goodai-ltm

## Quick start

    from goodai.ltm.memory_models.auto import AutoTextMemory

    mem = AutoTextMemory.create()
    mem.add_text("Lorem ipsum dolor sit amet, consectetur adipiscing elit\n")
    mem.add_text("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore\n",
                 metadata={'timestamp': '2023-04-19', 'type': 'generic'})
    r_memories = mem.retrieve(query='dolorem eum fugiat quo voluptas nulla pariatur?', k=3)
    for r_mem in r_memories:
        print(r_mem)

## Loading an embedding model

Recommended:

    em = AutoTextEmbeddingModel.from_pretrained('default-p4')

One embedding per passage:

    em = AutoTextEmbeddingModel.from_pretrained('default-p1')

Huggingface SentenceTransformer embeddings:

    em = AutoTextEmbeddingModel.from_pretrained('st:sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

OpenAI embeddings:

    em = AutoTextEmbeddingModel.from_pretrained('openai:text-embedding-ada-002')

## Embedding model usage

## Loading a query-passage matching model

Recommended:

    tmm = AutoTextMatchingModel.from_pretrained('default')

Huggingface reranking cross-encoders from the sentence-transformers library:

    tmm = AutoTextMatchingModel.from_pretrained('st:cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

## Query-passage matching model usage

## Loading a text memory instance:

Recommended:

    mem = AutoTextMemory.create()

Specify which models to use:

    tok = AutoTokenizer.from_pretrained('gpt2')
    config = TextMemoryConfig()
    config.chunk_capacity = 30  # tokens
    config.queue_capacity = 10000  # chunks
    vector_size = em.get_embedding_dim()
    faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(vector_size))
    mem = AutoTextMemory.create(emb_model=em,
        matching_model=None, tokenizer=tok,
        vector_db=faiss_index, config=config,
        device=torch.device('cuda:0'))

## Text memory usage

## Architecture

## Standard benchmarks

## Our evaluations
