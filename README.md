## GoodAI-LTM

GoodAI-LTM brings together all the components necessary for equipping agents with text-based long term memory. 
This includes text embedding models, reranking, vector databases, chunking, metadata such as time stamps and 
document information, memory and query rewriting (expansion and disambiguation), storage and retrieval. 

The package is especially adapted to provide a dialog-centric memory stream for social agents.

GoodAI-LTM also provides a conversational agent component (`LTMAgent`) that can be easily integrated into
Python-based apps.


## Installation

    pip install goodai-ltm

## Usage of LTMAgent


## Text memory: basic usage

The following code snippet creates an instance of LTM, loads in some text and then retrieves the most relevant text chunks given a query:

    from goodai.ltm.mem.auto import AutoTextMemory
    mem = AutoTextMemory.create()
    mem.add_text("Lorem ipsum dolor sit amet, consectetur adipiscing elit\n")
    mem.add_text("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore\n",
                 metadata={'title': 'My document', 'tags': ['latin']})
    r_memories = mem.retrieve(query='dolorem eum fugiat quo voluptas nulla pariatur?', k=3)

### Loading a text memory instance

A default memory instance can be created as follows:

    from goodai.ltm.mem.auto import AutoTextMemory

    mem = AutoTextMemory.create()

You can also configure the memory by passing parameters to the `create` method.
In the following example, the memory uses a "gpt2" tokenizer
for chunking, a T5 model for embeddings, a 
FAISS index for embedding storage instead of a simple vector
database, and a custom chunking configuration.

    import torch
    from transformers import AutoTokenizer
    from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
    from goodai.ltm.mem.auto import AutoTextMemory
    from goodai.ltm.mem.config import TextMemoryConfig
    from goodai.ltm.mem.mem_foundation import VectorDbType
    
    embedding_model = AutoTextEmbeddingModel.from_pretrained('st:sentence-transformers/sentence-t5-base')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    config = TextMemoryConfig()
    config.chunk_capacity = 30  # tokens
    config.queue_capacity = 10000  # chunks
    mem = AutoTextMemory.create(emb_model=embedding_model,
                                matching_model=None, 
                                tokenizer=tokenizer,
                                vector_db_type=VectorDbType.FAISS_FLAT_L2, 
                                config=config,
                                device=torch.device('cuda:0'))

### Adding text to memory

Call the `add_text` method to add text to the memory.
Text may consist of phrases, sentences or documents.

    mem.add_text("Lorem ipsum dolor sit amet, consectetur adipiscing elit\n")

Internally, the memory will chunk and index the text
automatically.

Text can be associated with an arbitrary metadata dictionary, such as:

    mem.add_text("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore\n",
                 metadata={'title': 'My document', 'tags': ['latin']})

Internally, the memory concatenates text stored using add_text with any text previously sent to the memory.

To retrieve a list of passages associated with a query,
call the `retrieve` method:

    r_memories = mem.retrieve(query='dolorem eum fugiat quo voluptas nulla pariatur?', k=3)

The `retrieve` method returns a list of objects of type `RetrievedMemory`, containing
the following properties:

* `passage`: The text of the memory. This corresponds to text found in a matching chunk, but it may be expanded using text from adjacent chunks.
* `timestamp`: The time (seconds since Epoch by default) when the retrieved chunk was created. 
* `distance`: Calculated distance between the query and the chunk passage.
* `confidence`: If a query-passage matching model is available, this is the probability assigned by the model.
* `metadata`: Metadata associated with the retrieved text, if any.

### Retrieval 

To retrieve memories, we pass a query and the desired number of memories to the method `retrieve`. For example,

    mem.retrieve("What does Jake propose?", k=2)

will return the two passages most relevant to the query.

The embedding model converts the query into an embedding. Then the stored embeddings closest to the query embedding 
are found and the corresponding texts decoded.

Optionally, a query-passage matching model can be used to compute pairwise query-passage matching probabilities 
and rerank the passages.

![Storage](diagram-simple.png)

## Embedding models

### Loading

An embedding model is loaded as follows:

    from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel

    em = AutoTextEmbeddingModel.from_pretrained(model_name)

The `model_name` can be one of the following:

* A SentenceTransformer (Huggingface), starting with `"st:"`, for example, `"st:sentence-transformers/multi-qa-mpnet-base-cos-v1"`.
* An OpenAI embedding model name, starting with `"openai:"`, for example, `"openai:text-embedding-ada-002"`.
* One of our fine-tuned models:

Name | Base model                                       | # parameters | # storage emb
---- |--------------------------------------------------|--------------| -----
em-MiniLM-p1-01 | multi-qa-MiniLM-L6-cos-v1 | 22.7m        | 1  
em-MiniLM-p3-01 | multi-qa-MiniLM-L6-cos-v1 | 22.7m        | 3  
em-distilroberta-p1-01 | sentence-transformers/all-distrilroberta-v1 | 82.1m        | 1
em-distilroberta-p3-01 | sentence-transformers/all-distrilroberta-v1 | 82.1m        | 3
em-distilroberta-p5-01 | sentence-transformers/all-distrilroberta-v1 | 82.1m        | 5

### Usage of embedding models

To get embeddings for a list of queries, call 
the `encode_queries` method, as follows:

    r_emb = em.encode_queries(['hello?'])

This returns a numpy array. To get a Pytorch tensor,
add the `convert_to_tensor` parameter:

    r_emb = em.encode_queries(['hello?'], convert_to_tensor=True)

To get embeddings for a list of passages, call 
the `encode_corpus` method, as follows:

    s_emb = em.encode_corpus(['it was...', 'the best of...'])

A peculiarity of our embedding model is that queries
and passages can have more than one embedding.
Embedding tensors have 3 axes: The batch size, the number of
embeddings, and the number of embedding dimensions. Typically,
the number of embeddings per query/passage will be 1, except for the 
passage embeddings in some of our fine-tuned models.

## Query-passage matching models

### Loading

A query-passage matching/reranking model can be loaded as follows:

    from goodai.ltm.reranking.auto import AutoTextMatchingModel
    
    model = AutoTextMatchingModel.from_pretrained(model_name)

The `model_name` can be one of the following:

* A "st:" prefix followed by the name of a Huggingface cross-encoder compatible with the SentenceTransformers library, like "st:cross-encoder/stsb-distilroberta-base"
* An "em:" prefix followed by the name of an embedding model supported by this library, like "em:openai:text-embedding-ada-002" or "em:em-distilroberta-p3-01"

Memory instances, by default, do not use a query-passage matching model. To enable one, it should be configured
as follows:

    from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
    from goodai.ltm.mem.auto import AutoTextMemory
    from goodai.ltm.mem.config import TextMemoryConfig
    from goodai.ltm.reranking.auto import AutoTextMatchingModel
    
    
    # Low-resource embedding model
    emb_model = AutoTextEmbeddingModel.from_pretrained('em-MiniLM-p1-01')
    # QPM model that boosts retrieval accuracy
    qpm_model = AutoTextMatchingModel.from_pretrained('em:em-distilroberta-p5-01')
    config = TextMemoryConfig()
    config.reranking_k_factor = 8
    mem = AutoTextMemory.create(matching_model=qpm_model, emb_model=emb_model, config=config)

The `reranking_k_factor` setting tells the memory how many candidates it should consider
for reranking. The user requests `k` memories. The reranking algorithm considers
`k * reranking_k_factor` chunks.

### Usage of query-passage matching models

The `predict` method of the model takes a list of
query-passage tuples and returns a list of floats
representing estimated match probabilities. Example:

    model = AutoTextMatchingModel.from_pretrained('em:em-distilroberta-p5-01')
    sentences = [
        ('Mike: What is your favorite color?', 'Steve: My favorite color is purple.'),
        ('Name the inner planets.', 'It was the best of times, it was the worst of times.'),
    ]
    prob = model.predict(sentences)
    print(prob)

## Embedding model evaluations

See the [evaluations README](./evaluations).

## More examples

Additional example code can be found in the `examples` folder. 

`examples/dump_mem.py` adds text to memory and shows how it is stored.

`examples/wiki_retrieval.py` stores and queries articles from Wikipedia.

`examples/rewriting.py` demonstrates query and memory rewriting.

## Use in GoodAI's AI game

An early application of GoodAI-LTM is in GoodAI's forthcoming [AI Game](https://www.goodai.com/ai-in-games/). 
LLMs are used to shape NPC behavior and to generate dialog. Long-term memory is used to provide characters 
with backstories and allows them to accumulate  experience during game play. 

[![AI-game](game-screenshot-25.png)](https://www.youtube.com/watch?v=xkn0H_iWDEQ)

Internally, we integrate the library with its C# port which allows us to split the library functionality between
server and client more easily. The C# part of the library has not been made public yet.

## Future plans

We will continue to improve GoodAI-LTM. Possible next steps include
* Retrieval weighted by recency and importance
* Flag for preventing internal text concatenation in mem.add_text()
* Embeddings for source code retrieval
* Storage and retrieval methods without embeddings
* Improvements to the currently experimental query and memory rewriting feature and its default prompts
* Configurable chunk overlapping and expansion
* Iterating on improvements to our datasets and models
