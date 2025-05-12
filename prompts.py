def get_query_prompt(question):
    prompt_template =f"""
        You are a chatbot that responds using the context provided. 
        You have to answer like a chatbot but only in the tone as present in the document. Follow the below steps:
        1. Answer the question strictly using only based on the given images.
        2. Ignore any image which does not contains information related to question asked.
        3. Answer should be informative.
        4. Give output in bullet points whenever possible.
        5. Give the reference at the end of the answer with the topic picked with pages names or page titles of the documents.
        6. In output do not mention shown in the image or according to image, image should not be there in answer.
        Question: {question}
        """
    return prompt_template



def get_output_guard_prompt(question, result):
    # Define limits
    prompt_template = f"""
    You are a safeguard bot who only gives "Allowed" or "Not allowed".
    You must judge the questions strictly according to the rules below, specifically in the context of a bank's case study involving challenges and solutions.
    
    Question: {question}  
    If the result given is irrelevant to the bank's case study or breaks the defined rules, reply "Not allowed".  
    Result: {result}
    
    Definitions (Context: Bank Case Study - Challenges and Solutions):
    
    User asks political or regulatory speculation:
    Examples:
        - "What is your view on the current government policies?"
        - "Who is responsible for the economic crisis?"
        - "Is the central bank acting politically?"
    
    User asks out of context (not related to the bank or the case study):
    Examples:
        - "How do airplanes fly?"
        - "What is the meaning of life?"
        - "What's the weather today?"
        - "What is your favorite movie?"
    
    User expresses insult or rudeness:
    Example:
        - "Your system is garbage."
    
    User asks to predict or forecast:
    Examples:
        - "How will the bank perform in the next quarter?"
        - "What’s the future growth of this bank?"
        - "Will this crisis happen again?"
    
    User seeks financial/legal advice:
    Examples:
        - "Should I invest in this bank?"
        - "Is this bank a good option for my loan?"
        - "What is the best account to open now?"
    
    User gives personal details (privacy concern):
    Examples:
        - "My account number is XXXX"
        - "My email is abc@domain.com"
    
    User references unrelated entities (e.g., celebrities, gossip):
    Examples:
        - "What does Elon Musk think about this bank?"
        - "Was a celebrity involved in the bank's crisis?"
    
    Responses:
    
    Bot answers political or speculative questions: → "Not allowed"  
    Bot answers out-of-context queries: → "Not allowed"  
    Bot answers prediction or forecasting requests: → "Not allowed"  
    Bot provides investment or legal advice: → "Not allowed"  
    Bot answers with or accepts personal/private info: → "Not allowed"  
    Bot engages with unrelated topics or celebrity content: → "Not allowed"  
    Bot receives insult: → "Let's keep the conversation respectful."
    
    Implementation:
    
    If the user query violates any of the above rules, respond strictly with "Not allowed".
    Focus only on questions related to the case study of the bank, such as:
        - The nature of the challenges the bank faced
        - The strategies or tools used to overcome them
        - Internal process changes or risk management improvements
        - Outcomes or lessons learned
    """
    return prompt_template




def check_initial_safeguard(question: str, result: str) -> str:
    political_keywords = [
        "political beliefs", "president", "government policies", "central bank acting politically",
        "left wing", "right wing", "economic crisis"
    ]
    
    out_of_context_keywords = [
        "drink tea", "air or water", "weather", "meaning of life", "favorite movie",
        "airplanes fly"
    ]

    insult_keywords = ["stupid", "garbage", "useless"]

    forecast_keywords = [
        "perform in near future", "return of my investment", "value after 2 years",
        "growth of this bank", "future trends", "where is the economy heading",
        "happen again"
    ]

    advice_keywords = [
        "should I invest", "good option for loan", "best account to open",
        "this stock will benefit you"
    ]

    stock_keywords = [
        "XYZ stock", "ABC stock", "good buy"
    ]

    personal_info_keywords = ["@", "account number", "contact me at"]

    celebrity_keywords = ["Salman Khan", "Elon Musk", "Ratanlal Tata"]

    mutual_fund_keywords = [
        "mutual fund", "government interest rates", "political affiliation",
        "budget impact mutual funds", "guarantee profits mutual funds",
        "regulated to ensure", "mutual funds downturn"
    ]

    all_block_keywords = political_keywords + out_of_context_keywords + forecast_keywords + \
        advice_keywords + stock_keywords + personal_info_keywords + celebrity_keywords + mutual_fund_keywords

    lowered_question = question.lower()

    for word in all_block_keywords:
        if word.lower() in lowered_question:
            return "Not allowed"

    for word in insult_keywords:
        if word.lower() in lowered_question:
            return "Let's keep the conversation respectful."

    return "Allowed"



def text_translate(query):

    prompt = f"""You are a helpful assistant that translates given language to english language only
                 your task is to just translate given text in english language only not to answer the question 
                 just return translated text and don't return any comment or explanation
                 and don't return Here is the English translation:
                 
                 if the given text is in english just return the given text
        
                 user query: {query} """
    return prompt



def classify_and_rephrase_query(query, recent_queries, bedrock):
    if not recent_queries:
        return "Classification: New query\nRephrased Query: " + query

    prompt = f"""
    You are an advanced AI assistant specializing in query analysis and contextualization. Your primary task is to strictly classify the current query as either a follow-up or a new query.

    Historical Conversation: {recent_queries}

    Current query: {query}

    Instructions:
    1. Carefully analyze the current query in relation to the recent queries.
    2. If it is talking about new fund, new scheme, it should be "New query".
    3. Strictly classify the query as either "Follow-up" or "New query".
    3. Based on your classification:
    a. If it's a "New query":
        - Pass the query exactly as it is, without any modifications.
    b. If it's a "Follow-up":
        - Identify all relevant context from recent queries.
        - Rephrase the current query to incorporate this context explicitly.
        - Ensure the rephrased query is comprehensive and self-contained and should not take any unnecessary and extra context from recent queries.
        - Make the rephrased query as crisp and context-rich as possible to enhance RAG retrieval accuracy.
        - Include specific details, entities, or concepts mentioned in previous queries if relevant.

    4. Use the following format for your response:

    Classification: [Follow-up or New query]
    Rephrased Query: [Your rephrased query here if it's a follow-up, or the original query if it's new]

    Note:
    - Be very strict in your classification. If there's any doubt, classify as "New query".
    - For new queries, do not modify the original query in any way.
    - Only rephrase and add context for queries classified as "Follow-up".
    """
    
    return prompt