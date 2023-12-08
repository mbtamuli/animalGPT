from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_path = "./film_reviews_fine_tuned_v2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# Initialize conversation history
conversation_history = ""

@app.route('/', methods=['GET', 'POST'])
def index():
    global conversation_history

    if request.method == 'POST':
        user_input = request.form['user_input']

        if user_input:
            # Combine user input with conversation history
            input_text = user_input

            # Generate response
            response = generate_text(input_text)


            return render_template('index.html', user_input=user_input, response=response, conversation_history=conversation_history)

    return render_template('index.html', user_input="", response="", conversation_history=conversation_history)

def generate_text(input_text):
    text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    generated_response = text_generator(input_text, max_length=150, num_return_sequences=2, temperature=0.5)[0]['generated_text']
    return generated_response

if __name__ == "__main__":
    app.run(debug=True,port=9810)
