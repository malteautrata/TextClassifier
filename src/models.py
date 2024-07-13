from transformers import T5Model, T5EncoderModel, AutoTokenizer, BertModel
from torch import nn
import torch


class T5Classifier(nn.Module):
    def __init__(self, loss_fn, lr, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        self.loss_fn = loss_fn
        self.lr = lr
        self.is_transformer = True
        if "use_gradient_clip" in kwargs:
            self.use_gradient_clip = kwargs.get("use_gradient_clip")
        else:
            self.use_gradient_clip = False
        self.t5_model = T5Model.from_pretrained("google-t5/t5-base")
        self.linear1 = nn.Linear(6912, 512)
        self.linear2 = nn.Linear(512, 9)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)

    def forward(
        self,
        input_ids,
        input_attention_mask,
        decoder_ids=None,
        decoder_attention_mask=None,
    ):
        out = self.t5_model(
            input_ids, input_attention_mask, decoder_ids, decoder_attention_mask
        ).last_hidden_state
        out = out[:, :, 0]
        return out


class T5EncoderClassifier(nn.Module):
    def __init__(self, loss_fn, lr, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")
        self.loss_fn = loss_fn
        self.lr = lr
        self.is_transformer = False
        if "use_gradient_clip" in kwargs:
            self.use_gradient_clip = kwargs.get("use_gradient_clip")
        else:
            self.use_gradient_clip = False

        self.t5encoder_model = T5EncoderModel.from_pretrained("google-t5/t5-large")
        self.linear1 = nn.Linear(524288, 512)
        self.linear2 = nn.Linear(512, 9)
        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = torch.flatten(
            self.t5encoder_model(input_ids, attention_mask).last_hidden_state,
            start_dim=1,
        )
        out = torch.tanh(out)
        out = self.dropout1(out)
        out = self.linear1(out)
        out = self.dropout2(out)
        out = torch.tanh(out)
        out = self.linear2(out)
        return out


class BertTextClassifier(nn.Module):
    def __init__(self, loss_fn, lr, **kwargs):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
        self.loss_fn = loss_fn
        self.lr = lr
        self.is_transformer = False
        if "use_gradient_clip" in kwargs:
            self.use_gradient_clip = kwargs.get("use_gradient_clip")
        else:
            self.use_gradient_clip = False

        self.bert_model = BertModel.from_pretrained("deepset/gbert-large")
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 9)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.bert_model(input_ids, attention_mask).pooler_output
        out = self.dropout(self.linear1(out))
        out = self.linear2(out)
        return out
