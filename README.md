## GoodAI-LTM

GoodAI-LTM equips agents with text-based long-term memory by combining essential components such as 
text embedding models, reranking, vector databases, memory and query rewriting, automatic chunking, 
chunk metadata, and chunk expansion. This package is specifically designed to offer a dialog-centric 
memory stream for social agents.

Additionally, GoodAI-LTM includes a conversational agent component (LTMAgent) for seamless 
integration into Python-based apps.

## Installation

    pip install goodai-ltm

## Usage of LTMAgent

Call the `reply` method of an `LTMAgent` instance to get a response from the agent.

    from goodai.ltm.agent import LTMAgent
    
    agent = LTMAgent(model="gpt-3.5-turbo")
    response = agent.reply("What can you tell me about yourself?")
    print(response)

The `model` parameter can be the name of any model supported by the [litellm library](https://github.com/BerriAI/litellm).

A session history is maintained automatically by the agent. If you want to start a 
new session, call the `new_session` method.

    agent.new_session()
    print(f"Number of messages in session: {len(agent.session.message_history)}")    

The agent has a conversational memory and also a knowledge base. You can tell the agent
to store knowledge by invoking the `add_knowledge` method.

    agent.clear_knowledge()
    agent.add_knowledge("The user's birthday is February 10.")
    agent.add_knowledge("Refer to the user as 'boss'.")
    response = agent.reply("Today is February 10. I think this is an important date. Can you remind me?")
    print(response)

`LTMAgent` is a seamless RAG system. The [ltm_agent_with_wiki](./examples/ltm_agent_with_wiki.py) example 
shows how to add Wikipedia articles to the agent's knowledge base.

You can persist the agent's configuration and its memories/knowledge by obtaining
its state as a string via the `state_as_text` method.

    state_text = agent.state_as_text()
    # Persist state_text to secondary storage

To build an agent from state text, call the `from_state_text` method.

    agent2 = LTMAgent.from_state_text(state_text)

Note that this does not restore the conversation session. The persist the conversation session
call the `state_as_text` method of the session.

    from goodai.ltm.agent import LTMAgentSession
    
    session_state_text = agent.session.state_as_text()
    # session_state_text can be persisted in secondary storage
    # The session.session_id field can serve as an identifier of the persisted session
    # Now let's restore the session in agent2
    p_session = LTMAgentSession.from_state_text(session_state_text)
    agent2.use_session(p_session)

## Usage of text memory (low level)

The following code snippet creates an instance of the LTM, loads in some text and then retrieves 
the most relevant text passages (expanded chunks) given a query:

    from goodai.ltm.mem.auto import AutoTextMemory
    mem = AutoTextMemory.create()
    mem.add_text("Lorem ipsum dolor sit amet, consectetur adipiscing elit\n")
    mem.add_text("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore\n",
                 metadata={'title': 'My document', 'tags': ['latin']})
    r_memories = mem.retrieve(query='dolorem eum fugiat quo voluptas nulla pariatur?', k=3)

### Creating a text memory instance

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

The memory concatenates text stored using `add_text` with any text previously sent to the memory,
but you can call `add_separator` to ensure that new text is not added to previously created chunks.

To retrieve a list of passages associated with a query,
call the `retrieve` method:

    r_memories = mem.retrieve(query='dolorem eum fugiat quo voluptas nulla pariatur?', k=3)

The `retrieve` method returns a list of objects of type `RetrievedMemory`, in descending order of
relevance. Each retrieved memory has the following properties:

* `passage`: The text of the memory. This corresponds to text found in a matching chunk, but it may be expanded using text from adjacent chunks.
* `timestamp`: The time (seconds since Epoch by default) when the retrieved chunk was created. 
* `distance`: Calculated distance between the query and the chunk passage.
* `relevance`: A number between 0 and 1 representing the relevance of the retrieved memory.
* `confidence`: If a query-passage matching model is available, this is the probability assigned by the model.
* `metadata`: Metadata associated with the retrieved text, if any.

### Retrieval 

To retrieve memories, we pass a query and the desired number of memories to the method `retrieve`. For example,

    mem.retrieve("What does Jake propose?", k=2)

will return the two passages most relevant to the query.

The embedding model converts the query into an embedding. Then the stored embeddings closest to the query embedding 
are found and the corresponding texts decoded.

## Embedding models

### Loading

An embedding model is loaded as follows:

    from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel

    em = AutoTextEmbeddingModel.from_pretrained(model_name)

The `model_name` can be one of the following:

* A SentenceTransformer (Huggingface), starting with `"st:"`, for example, `"st:sentence-transformers/multi-qa-mpnet-base-cos-v1"`.
* A flag embedding model, starting with `"flag:"`, for example, `"flag:BAAI/bge-base-en-v1.5"`.
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

Queries and passages can have more than one embedding.
Embedding tensors have 3 axes: The batch size, the number of
embeddings, and the number of embedding dimensions. Typically,
the number of embeddings per query/passage will be 1, with some exceptions.

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

## Agent benchmarks

Refer to the [goodai-ltm-benchmark project page](https://github.com/GoodAI/goodai-ltm-benchmark).

## More examples

Additional example code can be found in the `examples` folder. 
