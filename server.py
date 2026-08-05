import os
from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient
from supabase import create_client, Client

app = Flask(__name__, template_folder='.')

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
HF_KEY = os.environ.get("HF_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
chatClient = InferenceClient(
    base_url="https://router.huggingface.co/v1",
    token=HF_KEY,
)
embeddingClient = InferenceClient(
    provider="hf-inference",
    token=HF_KEY,
)
def getEmbedding(text):
    result = embeddingClient.feature_extraction(
        text,
        model="BAAI/bge-small-en-v1.5",
        normalize=True,
    )
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
        return result[0]
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def askBot():
    data = request.get_json(silent=True) or {}
    userQuestion = data.get("question", "")
    
    try:
        queryVector = getEmbedding(userQuestion)
        db_response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": queryVector,
                "match_threshold": 0.4,
                "match_count": 7
            }
        ).execute()

        matchedFacts = db_response.data
        context_str = ""
        if matchedFacts:
            context_str = "\n".join([f"- {fact['content']}" for fact in matchedFacts])

        system_instruction = (
            "[CONTEXT]:\n"
            "You are a respectful, welcoming educational assistant for the Fremont Gurdwara. "
            "Your job is to teach visitors about Gurdwara etiquette (like removing shoes, covering the head) and Sikh history.\n\n"

            "[ROLE]:\n"
            "An educational assistant for the Fremont Gurdwara.\n\n"

            "[OBJECTIVE]:\n"
            "To teach visitors about Gurdwara etiquette and Sikh history using the verified database data.\n\n"

            "[FORMAT]:\n"
            "- Use bullet points whenever it is easier for the user to understand and read.\n"
            "- Insert a double blank line break before your closing follow-up question so it sits on a brand-new paragraph below the bullet points.\n\n"

            "[TONE]:\n"
            "Keep your tone warm and highly welcoming. Stay respectful.\n\n"

            "[CONSTRAINTS]:\n"
            "- VERY VERY Important: make sure to not have such long answers(try to keep under 7 sentences).\n"
            "- Do not append generic phrases like 'How can I assist you today?' or 'What would you like to know?' if you already asked a contextual question.\n"
            "- Only answer questions related to these topics. If a user asks an unrelated question, politely decline.\n"
            "- Since the starting message already asks the question about choosing a language, if they choose to IGNORE it DO NOT ask them again.\n"
            "- CRITICAL: IF THE USER CHOOSES A LANGUAGE YOU HAVE TO STICK TO IT AND DO NOT EVER ASK THEM AGAIN IF THEY WANT TO CHANGE IT\n"
            "- There is no need to welcome the user again after the first time!\n"
            "- If a user types random letters or you dont understand, ask to rephrase. DO NOT list the etiquette rules or bullet points in this scenario.\n"
            "- IF AND ONLY IF the user simply says 'Hi' or 'hello' or a greeting, respond with a warm greeting + ask how you can help them. DO NOT LIST RULES IN THIS CASE EITHER.\n\n"

            "VERIFIED WEBSITE DATABASE DATA:\n"
            f"{context_str if context_str else '- No specific data found.'}"
        )


        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": userQuestion}
        ]
        
        response = chatClient.chat_completion(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=messages,
            max_tokens=300,
            temperature=0.0
        )
        
        reply = response.choices[0].message.content
        return jsonify({"reply": reply.strip()})

    except Exception as e:
        print(f"BACKEND ERROR LOG: {e}")
        return jsonify({"reply": f" backend error: {str(e)}"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
