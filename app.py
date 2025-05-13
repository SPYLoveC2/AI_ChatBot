import pandas as pd
from fastapi import FastAPI, Request
from datetime import datetime
from pydantic import BaseModel
from execute import get_answer


app = FastAPI()

history_path = "./chat_history/chat_history.csv"
history = {
    "question_asked":[],
    "classification":[],
    "classified_question":[],
    "chat_time":[]
}
df = pd.DataFrame(history)
df.to_csv(history_path, index=False)


class ReqeustValidator(BaseModel):
    question:str


def get_chat_history(n):
    history = pd.read_csv(history_path)
    history['chat_time'] = pd.to_datetime(history['chat_time'])
    history.sort_values("chat_time", ascending=False)
    history = history.head(n)['question_asked'].values
    string = ""
    for i, hist in enumerate(history):
        string+= f"{n-i}. {hist}\n\n"
    return string



def save_history(question, classified_query, query_type):
    history = pd.read_csv(history_path)
    new_row = pd.DataFrame([{'question_asked': question, 
                             'classification': query_type,
                             'classified_question': classified_query,
                             'chat_time': datetime.now()}])
    
    history = pd.concat([history, new_row], ignore_index=True)
    history.to_csv(history_path, index=False)



@app.api_route("/chat", methods=['GET', 'POST'])
def get_response(data: ReqeustValidator, request:Request):
    method = request.method
    print(method)
    if method=='GET':
        return {
            "status": "failed",
            "status code": 405,
            "Detail": "Get Method Not allowed"
        }
    question = data.question
    history = get_chat_history(3)
    response = get_answer(question, history)
    if len(response)>1:
        response, classified_query, query_type = response
        save_history(question, classified_query, query_type)
    else:
        return {
        "status": "failed",
        "status code": 400,
        "response": "Invalid query"

    }
    return {
        "status": "success",
        "status code": 200,
        "response": response

    }