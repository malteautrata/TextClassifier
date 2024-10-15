from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from scripts.load_dataset import label_to_id

class InstructModelWrapper:
    def __init__(self, **model_kwargs) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_kwargs["path"],
            torch_dtype=model_kwargs["torch_dtype"],
            device_map=model_kwargs["device_map"],
            quantization_config=model_kwargs["quantization_config"]
            if "quantization_config" in model_kwargs
            else None,
            use_cache=model_kwargs["use_cache"]
            if "use_cache" in model_kwargs
            else None,  # set to False as we're going to use gradient checkpointing
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_kwargs["tokenizer_path"])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.truncated_count = 0
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        self.system_message = """Kategorisiere die Eingabe in eine der folgendenen Kategorien. Antworte nur in einer der folgenden Kategorien:
        - Web
        - International
        - Etat
        - Wirtschaft
        - Panorama
        - Sport
        - Wissenschaft
        - Kultur
        - Inland"""

    def create_test_messages(self, sample: dict) -> dict:
        if len(sample["text"]) > 1024:
            sample["text"] = sample["text"][:1024]
            self.truncated_count += 1
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": sample["text"]},
        ]
        return {"messages": messages}

    def create_train_messages(self, sample: dict) -> dict:
        if len(sample["text"]) > 1024:
            sample["text"] = sample["text"][:1024]
            global truncated_count
            self.truncated_count += 1
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": sample["text"]},
            {"role": "assistant", "content": sample["label"]},
        ]

        return {"messages": messages}

    def tokenize_messages(self, sample: dict) -> dict:
        return {
            "input_ids": self.tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        }

    def apply_chat_template(self, sample: dict) -> dict:
        return {
            "chat_template": self.tokenizer.apply_chat_template(
                sample["messages"], tokenize=False
            )
        }


class HeadClassifierWrapper:
    def __init__(self, **model_kwargs) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_kwargs["path"],
            num_labels=model_kwargs["num_labels"],
            quantization_config=model_kwargs["quantization_config"],
            device_map=model_kwargs["device_map"],
            use_cache=model_kwargs["use_cache"]
            if "use_cache" in model_kwargs
            else None,  # set to False as we're going to use gradient checkpointing
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_kwargs["tokenizer_path"])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_text(self, sample):
        return self.tokenizer(
            sample["text"], max_length=1500, truncation=True, padding=True
        )

    def add_label_id(self, example):
        return {"label": label_to_id[example["label"]]}
