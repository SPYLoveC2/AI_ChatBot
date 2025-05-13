import os
import dotenv
import pandas as pd
from google import genai
from google.genai import types

from prompts import (get_query_prompt, get_output_guard_prompt, 
                     check_initial_safeguard, text_translate, classify_and_rephrase_query)
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
    payload = [[question]]
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



def get_classified_query(translated_question, history):
    classified_query = get_llm_response(classify_and_rephrase_query(query=translated_question, 
                                                                    recent_queries=history))
    print("="*30)
    print(f"Classified raw query: {classified_query}")
    stage1 = classified_query.split("\n")
    stage2 = stage1[0].split(':')

    try:
        print(stage2[1])
        if 'follow' in stage2[1].lower():
            return stage1[1].split(':')[1], 'follow'
        else:
            return translated_question, 'new'
    except Exception as e:
        print(f"Exception Occured: {e}")
        return translated_question, 'new'



def get_answer(question, history):
    print("="*30)
    print(f"Question: {question}  \n\nHistory: {history}")

    if question.lower() in ['hi', 'hello', 'hi, how are you?', 
                            'good morning', 'good evening', 'good night']:
        return "Hello, How can I help you today?"

    if check_initial_safeguard(question).lower()!='allowed':
        return ("No allowed", )

    translated_question = get_llm_response(text_translate(question))
    print("="*30)
    print(f"Translated question, {translated_question}")
    
    classified_query, query_type = get_classified_query(translated_question=translated_question, 
                                            history=history)
    
    print("="*30)
    print(f"Processed classified query, {classified_query}\nQuery Type: {query_type}")

    docs = get_documents(classified_query)
    response = get_questions_response(question=classified_query, docs=docs)

    print("="*30)
    print(f"Response: {response}")
    output_guard_result = get_llm_response(get_output_guard_prompt(question=classified_query, 
                                                                   result=response))
    print("="*30)
    print(f"Output Guard Result: {output_guard_result}")
    if 'not allowed' in output_guard_result.lower():
        return ("Not Allowed",)
    else:
        return response, classified_query, query_type
