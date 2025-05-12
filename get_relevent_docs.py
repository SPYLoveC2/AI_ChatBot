import os
import dotenv
from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus import connections, Collection
from google import genai
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


################################################ Initializing vars and connections ################################################
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)
connections.connect("default", host="localhost", port="19530")
collection = Collection(name='embedding_db')
collection.load()



################################################ Initializing Milvus builtin embedding model ##########################################
ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3',
    device='cpu',  # Change to 'cuda' if GPU support is required
    use_fp16=False
)



################################################ Function to get dense embedding of summary from google gemini ################################
def generate_dense_embeddings(text):
    result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text)
    return result.embeddings[0].values




################################################ Function to get sparse embedding of summary from milvus ################################
def generate_sparse_embeddings(text):
    global ef
    spark_embedding = ef([text])
    return spark_embedding['sparse']



def get_documents(query):
    dense_req = AnnSearchRequest(
    data = [generate_dense_embeddings(query)],
    anns_field='dense_vec',
    limit = 4,
    param={"metric_type": "IP"}
    )

    sparse_req = AnnSearchRequest(
    data = generate_sparse_embeddings(query),
    anns_field='sparse_vec',
    limit = 4,
    param={"metric_type": "IP"}
    )

    result = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=RRFRanker(),
        output_fields=['image_path', 'summary'],
        limit=3
        
    )
    return result