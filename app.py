from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model from local cache
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

sentiment_labels = {
    1: "Very Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Very Positive"
}

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            height: 100vh;
            margin: 0;
            display: flex;
            background: linear-gradient(
                to right,
                #2b2b2b 0%,
                #2b2b2b 74%,
                #141414 75%, /* darker blackish grey */
                #141414 100%
            );
            color: white;
        }
        .left {
            flex: 3;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .right {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        .container {
            background: white;
            color: black;
            padding: 60px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.5);
            text-align: center;
            width: 600px;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 20px;
            color: #333;
        }
        input[type="text"] {
            width: 90%;
            padding: 14px;
            font-size: 1.1rem;
            margin: 0 auto 15px auto;
            border-radius: 8px;
            border: 1px solid #ccc;
            outline: none;
            text-align: center;
            font-family: 'Times New Roman', serif;
            display: block;
        }
        input[type="text"]:focus {
            border-color: #d4af37;
        }
        button {
            background-color: black; /* black button */
            color: white;
            padding: 14px 30px;
            font-size: 1.1rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s ease;
            font-family: 'Times New Roman', serif;
        }
        button:hover {
            background-color: #d4af37; /* gold on hover */
            color: black;
        }
        .result {
            margin-top: 20px;
            font-size: 1.3rem;
            font-weight: bold;
            color: #444;
        }
        .recruit-title {
            font-size: 2rem;
            font-weight: bold;
            font-family: 'Times New Roman', serif;
            margin-bottom: 30px;
        }
        .qr-code {
            max-width: 200px;
            height: auto;
        }

        .mrm-logo {
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            width: 250px; /* size as before */
            height: 250px;
            border-radius: 50%; /* perfectly circular */
            object-fit: cover;
            background-color: transparent;
            box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        }

    </style>
</head>
<body>
    <div class="left">
        <div class="container">
            <h1>Sentiment Analysis</h1>
            <form method="POST">
                <input type="text" name="text" placeholder="Type your sentence..." value="{{ text }}" required>
                <button type="submit">Analyze</button>
            </form>
            {% if sentiment %}
                <div class="result">Result: {{ sentiment }}</div>
            {% endif %}
        </div>
    </div>
    <div class="right">
        <img src="{{ url_for('static', filename='mrm_logo.jpeg') }}" alt="MRM Logo" class="mrm-logo">
        <div class="recruit-title">JOIN MARS ROVER MANIPAL</div>
        <img src="{{ url_for('static', filename='qr.jpeg') }}" alt="QR Code" class="qr-code">
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    text = ""  # initialize so GET requests have a value
    if request.method == "POST":
        text = request.form["text"]
        tokens = tokenizer.encode(text, return_tensors='pt')
        result = model(tokens)
        score = int(torch.argmax(result.logits)) + 1
        sentiment = sentiment_labels[score]
    return render_template_string(HTML_TEMPLATE, sentiment=sentiment, text=text)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text") or request.json.get("text", "")
    tokens = tokenizer.encode(text, return_tensors='pt')
    result = model(tokens)
    score = int(torch.argmax(result.logits)) + 1
    return jsonify({"sentiment": sentiment_labels[score]})

if __name__ == "__main__":
    app.run(debug=True)
