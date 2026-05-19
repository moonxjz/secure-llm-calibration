import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("GenTelLab/gentelshield-v1")
model = ORTModelForSequenceClassification.from_pretrained("GenTelLab/gentelshield-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Conservative cap avoids ONNX shape expansion failures on long prompts.
MAX_LENGTH = 512

def pipeline(text: str) -> tuple[str, int, float]:
    """Return (label_str, label_int, attack_probability)."""
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()
            attack_prob = float(probs[0][1].item())

        label_map = {0: "safe", 1: "unsafe"}
        prediction = label_map[predicted_label]
        return prediction, predicted_label, attack_prob
    except Exception:
        # Fail-open to safe so the LLM stage can still classify.
        return "safe", 0, 0.5
        
