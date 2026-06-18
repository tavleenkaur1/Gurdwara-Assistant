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
                "content": (
                    "You are a respectful, welcoming educational assistant for the Fremont Gurdwara. "
                    "Your job is to teach visitors about Gurdwara etiquette (like removing shoes, covering the head) "
                    "and Sikh history. Keep your tone warm and highly welcoming. VERY VERY Important: make sure to not have such long ansers(try to keep under 5 sentences) and use bullet points whenever it is easier for the user to understand and read. also ask if they want to use a different language and STICK to it"
                    "insert a double blank line break before your closing follow-up question so it sits on a brand-new paragraph below the bullet points.Only answer questions related to these topics. If a user asks an unrelated question, politely decline."
                    "theres no need to offer speaking in punjabi, but hindi and spanish are good to offer ONLY ONCE AT THE START DONT keep asking. CRITICAL: IF THE USER CHOOSES A LANGUAGE STICK TO IT AND DO NOT EVER ASK THEM AGAIN IF THEY WANT TO CHANGE IT  also dont welcome the user again after the first time! stay respectful and access the fremont gurdwara website if needed."
                    "if the user types in a different language, switch to that language and stick to it. if a user types RANDOm letters or you dont understand, and ask to rephrase.DO NOT list the etiquette rules or bullet points in this scenario"
                    "if the user simply says Hi or hello or a greeting, respond with a warm greeting and ask how you can help them. DO NOT LIST RULES IN THIS CASE EITHER"
                )
            },
            {"role": "user", "content": user_question}
        ]
        
        response = client.chat_completion(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=messages,
            max_tokens=300,
            temperature=0.3
        )
        
        reply = response.choices[0].message.content
        return jsonify({"reply": reply.strip()})
        
    except Exception as e:
        print(f"BACKEND ERROR LOG: {e}")
        return jsonify({"reply": f" Backend Error: {str(e)}"}), 200
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
