from flask import Flask, request, jsonify
from scripts.encoder_model_wrapper import BertTextClassifier, T5EncoderClassifier
from scripts.llama_model_wrapper import HeadClassifierWrapper
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from scripts.load_dataset import id_to_label
import torch
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

device = "mps"
bert_model: BertTextClassifier = torch.load(
    "./results/bert_results/model.nosync/model.pt", map_location=device
)

t5_base_model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
    "./results/t5_results/base/model.nosync", device_map=device
)
t5_base_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

t5_encoder_model: T5EncoderClassifier = torch.load(
    "./results/t5_results/encoder/model.nosync/model.pt", map_location=device
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
llama_model_kwargs = {
    "path": "./results/llama3_results/classification_head/model.nosync/finetuned",
    "num_labels": 9,
    "tokenizer_path": "./results/llama3_results/classification_head/model.nosync/finetuned",
    "device_map": "auto",
    "use_cache": False,
    "quantization_config": quantization_config,
}
if device == "cuda":
    llama_model: HeadClassifierWrapper = HeadClassifierWrapper(**llama_model_kwargs) # only available on cuda


@app.route("/")
def hello_world():
    return("<p>Available classifiers: BERT, T5 Base, T5 Encoder, Llama 3 Classification Head")

@app.post("/bert_classifier")
def bert_classifier():
    """Endpoint for classification using the BERT model
    This is using docstrings for specifications.
    ---
    parameters:
      - name: body
        in: body
        schema:
          title: text
          type: object
          properties:
            text:
              type: string
              description: Input text
        example: 'Der 1. FC Köln hat heute 2:1 gegen den FC Bayern München gewonnen.'
        required: true
    responses:
      200:
        description: The category of the text.
        examples:
          text: 'Der FC Bayern hat heute gewonnen.'
          prediction: 'Sport'
    """
    input = request.get_json()
    output = {}
    text = input["text"]
    output["text"] = text

    model_input = bert_model.tokenizer(text=text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    prediction = bert_model(model_input["input_ids"].to(device=device), model_input["attention_mask"].to(device=device)
    )
    output["prediction"] = id_to_label[torch.argmax(prediction).item()]
    
    return jsonify(output), 200

@app.post("/t5_base_classifier")
def t5_base_classifier():
    """Endpoint for classification using the T5 base model
    This is using docstrings for specifications.
    ---
    parameters:
      - name: body
        in: body
        schema:
          title: text
          type: object
          properties:
            text:
              type: string
              description: Input text
        example: 'Der 1. FC Köln hat heute 2:1 gegen den FC Bayern München gewonnen.'
        required: true
    responses:
      200:
        description: The category of the text.
        examples:
          text: 'Der FC Bayern hat heute gewonnen.'
          prediction: 'Sport'
    """
    input = request.get_json()
    output = {}
    text = input["text"]
    output["text"] = text

    text = "Klassifiziere nachfolgenden Artikel in eine der folgendenen Kategorien: Web, International, Etat, Wirtschaft, Panorama, Sport, Wissenschaft, Kultur oder Inland:\n" + text
    model_input = t5_base_tokenizer(text=text, truncation=True, padding='max_length', max_length=512, return_tensors="pt").to(device=device)
    output["prediction"] = t5_base_tokenizer.batch_decode(t5_base_model.generate(**model_input, max_new_tokens=5), skip_special_tokens=True)
    return jsonify(output), 200

@app.post("/t5_encoder_classifier")
def t5_encoder_classfier():
    """Endpoint for classification using the T5 Encoder model
    This is using docstrings for specifications.
    ---
    parameters:
      - name: body
        in: body
        schema:
          title: text
          type: object
          properties:
            text:
              type: string
              description: Input text
        example: 'Der 1. FC Köln hat heute 2:1 gegen den FC Bayern München gewonnen.'
        required: true
    responses:
      200:
        description: The category of the text.
        examples:
          text: 'Der FC Bayern hat heute gewonnen.'
          prediction: 'Sport'
    """
    input = request.get_json()
    output = {}
    text = input["text"]
    output["text"] = text

    model_input = t5_encoder_model.tokenizer(text=text, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    prediction = t5_encoder_model(model_input["input_ids"].to(device=device), model_input["attention_mask"].to(device=device)
    )
    output["prediction"] = id_to_label[torch.argmax(prediction).item()]
    
    return jsonify(output), 200

@app.post("/llama_classifier")
def llama_classifier():
    """Endpoint for classification using the Llama 3 Classification Head model
    This is using docstrings for specifications.
    ---
    parameters:
      - name: body
        in: body
        schema:
          title: text
          type: object
          properties:
            text:
              type: string
              description: Input text
        example: 'Der 1. FC Köln hat heute 2:1 gegen den FC Bayern München gewonnen.'
        required: true
    responses:
      200:
        description: The category of the text.
        examples:
          text: 'Der FC Bayern hat heute gewonnen.'
          prediction: 'Sport'
    """
    input = request.get_json()
    output = {}
    text = input["text"]
    output["text"] = text

    model_input = llama_model.tokenizer(text, max_length=1500, truncation=True, padding=True, return_tensors="pt")
    prediction = llama_model.model(
        model_input["input_ids"].to(device=device),
        model_input["attention_mask"].to(device=device)
    ).logits
    output["prediction"] = id_to_label[torch.argmax(prediction).item()]
    return jsonify(output), 200

app.run()