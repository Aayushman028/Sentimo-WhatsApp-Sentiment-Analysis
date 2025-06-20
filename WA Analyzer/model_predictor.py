import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel, AutoConfig

class SentimentClassifier(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state[:, 0, :]      # CLS token
        dropped = self.dropout(hidden)
        return self.classifier(dropped)

class SentimentService:
    def __init__(self, model_dir: str):
        # Load tokenizer & config
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        config = AutoConfig.from_pretrained(model_dir)
        num_labels = config.num_labels
        self.id2label = {int(k): v for k, v in config.id2label.items()}

        # Build model & load weights
        self.model = SentimentClassifier(num_labels=num_labels)
        state = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, texts: list[str], batch_size: int = 64) -> list[dict]:
        """
        Processes texts in chunks for speed and lower memory.
        Returns list of {"label", "score"}.
        """
        results = []
        n = len(texts)
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            # tokenize batch
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=64,          # shorter sequences
                return_tensors="pt"
            )
            # move to device
            enc = {k: v.to(self.device) for k, v in enc.items()}

            # forward
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"]
                )  # (B, num_labels)
            probs = F.softmax(logits, dim=-1).cpu()

            # unpack
            for row in probs:
                idx = int(row.argmax().item())
                results.append({"label": self.id2label[idx], "score": float(row[idx].item())})

        return results
