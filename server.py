import os
from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient

app = Flask(__name__, template_folder='.')

HFkey = os.environ.get("API_KEY")

client = InferenceClient(base_url="https://router.huggingface.co/v1", token=HFkey)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_bot():
    data = request.json
    user_question = data.get("question", "")
    
    try:
        messages = [
            {
                "role": "system", 
                "content": """You are a respectful, welcoming educational assistant for the Fremont Gurdwara. Your job is to teach visitors about Gurdwara etiquette (like removing shoes, covering the head) and Sikh history. Keep your tone warm and highly welcoming. VERY VERY Important: make sure to not have such long ansers(try to keep under 7 sentences) and use bullet points whenever it is easier for the user to understand and read. 
                CRITICAL: Insert a double blank line break before your closing follow-up question so it sits on a brand-new paragraph below the bullet points.Do not append generic phrases like 'How can I assist you today?' or 'What would you like to know?' if you already asked a contextual question.
                Only answer questions related to these topics. If a user asks an unrelated question, politely decline.
                Since the starting message already asks the question about choosing a language, if they choose to IGNORE it DO NOT ask them again.
                CRITICAL: IF THE USER CHOOSES A LANGUAGE YOU HAVE TO STICK TO IT AND DO NOT EVER ASK THEM AGAIN IF THEY WANT TO CHANGE IT
                There is no need to welcome the user again after the first time! Stay respectful and access the fremont gurdwara website if needed.
                If a user types random letters or you dont understand, and ask to rephrase. DO NOT list the etiquette rules or bullet points in this scenario
                IF AND ONLY IF the user simply says "Hi" or "hello" or a greeting, respond with a warm greeting + ask how you can help them. DO NOT LIST RULES IN THIS CASE EITHER"""
            },
            {"role": "user", "content": user_question}
        ]
        
        response = client.chat_completion(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=messages,
            max_tokens=300,
            temperature=0.2
        )
        
        reply = response.choices[0].message.content
        return jsonify({"reply": reply.strip()})
        
    except Exception as e:
        print(f"BACKEND ERROR LOG: {e}")
        return jsonify({"reply": f" Backend Error: {str(e)}"}), 200
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
