import os
import dotenv
from google import genai, types
from prompts import (get_query_prompt, get_output_guard_prompt, 
                     check_initial_safeguard, text_translate)
from get_relevent_docs import get_documents
from pymilvus import connections, Collection



################################################ Initializing vars and connections ################################################
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)
connections.connect("default", host="localhost", port="19530")
collection = Collection(name='embedding_db')
collection.load()


def get_llm_response(prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text


def get_questions_response(question, docs):
    payload = [["What are challenges which are faced by kotak bank"]]
    for doc in docs[0]:
        image_path = doc['entity']['image_path']
        print(image_path)
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        img = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            )
        payload.append(img)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=payload
        )

    return response.text
    



def get_answer(question):
    if question.lower() in ['hi', 'hello', 'hi, how are you?', 
                            'good morning', 'good evening', 'good night']:
        return "Hello, How can I help you today?"

    if check_initial_safeguard(question).lower()!='allowed':
        return "No allowed"

    translated_question = get_llm_response(text_translate(question))
    print("==========================================================")
    print(question)
    print(translated_question)
    question = translated_question
    docs = get_documents(question)
    response = get_questions_response(question=question, docs=docs)



    




