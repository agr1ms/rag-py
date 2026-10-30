import chromadb
chroma_client = chromadb.Client()

collection_name = "agrim-collection"
collection = chroma_client.get_or_create_collection(collection_name)


# defining text document here, 3 documents
documents = [
    {"id":"doc1", "text":"Hi, Agrim!"},
    {"id":"doc1", "text":"Nice to see you"},
    {"id":"doc1", "text":"See ya more often!"},
]

# adding the documents into our collection. Imp- I am not going to use .add method, instead will use the upsert, as it will make sure that even if we rerun everything, it doesn't continously keep adding the same documents
for doc in documents:
    collection.upsert(ids=doc["id"], texts=doc["text"])

# this is the query that we are gonna be passing to chroma DB
# Then, the chroma db is going to create a little embedding go inside and look for the document that is the most similar
query_text = "Hi, Agrim!"

# get the results by running the collection query by passing teh query
results = collection.query(
    query_texts=[query_text],
    n_results=3,
)

print(results)
