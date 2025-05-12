import os
import dotenv
import os
import time
from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, list_collections, drop_collection
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from raw_data_splitter import get_files_path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')



################################################ Initializing vars and connections ################################################
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)
connections.connect("default", host="localhost", port="19530")
metadata = {"image_path": [], "summary": [], "dense_vector": [], "sparse_vector": []}

embedding_dim_dict = {
    'gemini-embedding-exp-03-07': 3072,
    'text-embedding-004': 768,
    'milvus_BGEM3Embedding': 1024
}

embedding_model = 'gemini-embedding-exp-03-07'
dense_vector_dim = embedding_dim_dict[embedding_model]

################################################ Initializing Milvus builtin embedding model ##########################################
ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3',
    device='cpu',  # Change to 'cuda' if GPU support is required
    use_fp16=False
)


################################################ Function to get summary of image from google gemini ######################################
def generate_summary(image):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            systemInstruction="You are an expert in analyzing document images. Your task is to provide detailed and accurate summaries for each page of the document.Highlight bankname for which this document is about. The summaries should capture all key topics, specific information, and any references to other pages. The summaries will be used to facilitate precise retrieval of relevant pages in a retrieval-augmented generation (RAG) system."
        ),
        contents=[image]
    )
    return response.text




################################################ Function to get dense embedding of summary from google gemini ################################
def generate_dense_embeddings(text, embd_model):
    if embd_model in ['gemini-embedding-exp-03-07', 'text-embedding-004']:
        result = client.models.embed_content(
                model=embd_model,
                contents=text)
        embedding = result.embeddings[0].values
    else:
        global ef
        embedding =ef([text])['dense'][0].tolist()
    return embedding



################################################ Function to get sparse embedding of summary from milvus ################################
def generate_sparse_embeddings(text):
    global ef
    sparse_embedding = ef([text])
    return sparse_embedding['sparse']



################################################ Defining collection schma ################################
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=10_000),
    FieldSchema(name="dense_vec", dtype=DataType.FLOAT_VECTOR, dim=dense_vector_dim),
    FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR)
]

schema = CollectionSchema(fields, description="schema defined to store embedding")
collection_name = "embedding_db"
if collection_name in list_collections():
    print("\n\n=====================================================")
    userinput = input(f"collection {collection_name} already exist, DELETE? YES/NO  ")     ### If collection already exist delete that collection
    if userinput.lower()=="yes":
        drop_collection(collection_name=collection_name)
        print(f"Collection: {collection_name} deleted")

collection = Collection(name=collection_name,schema=schema)



################################################ Generating embedding of each image's summary ################################
files_path = get_files_path("raw_data/Case Study Summary (1).pptx")    #### Generating images for each page from raw data and return paths of each image
for path in tqdm(files_path):
    image = Image.open(path)
    try:
        summary = generate_summary(image)
        sparse_embedding = generate_sparse_embeddings(summary)
        dense_embedding = generate_dense_embeddings(summary, embedding_model)
    except Exception as e:
        print(f"Error: {e} occured sleeping for 60 sec")
        print(datetime.now())
        time.sleep(60)
        print(datetime.now())
        summary = generate_summary(image)
        sparse_embedding = generate_sparse_embeddings(summary)
        dense_embedding = generate_dense_embeddings(summary, embedding_model)

    metadata['image_path'].append(path)
    metadata['summary'].append(summary)
    metadata['dense_vector'].append(dense_embedding)
    metadata['sparse_vector'].append(sparse_embedding)


collection.insert([metadata['image_path'], 
                   metadata['summary'], 
                   metadata['dense_vector'], 
                   metadata['sparse_vector']
                  ]
                 )

collection.flush()
collection.create_index("sparse_vec", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
collection.create_index("dense_vec", {"index_type": "FLAT", "metric_type": "IP"})

connections.disconnect('default')
