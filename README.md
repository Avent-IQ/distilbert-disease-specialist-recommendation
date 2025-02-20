# distilbert-disease-specialist-recommendation

## Model Overview
This is a **Zero-shot Classification Model** designed to classify the appropriate medical department or specialist a patient should consult based on their symptoms. The model is built using **DistilBERT** and trained for multi-class classification.

### **Supported Classes:**
- **Cardiology** (Heart-related symptoms)
- **Neurology** (Brain and nervous system issues)
- **Orthopedics** (Bone and muscle problems)
- **Dermatology** (Skin-related conditions)

## Training Data
The model is trained on a curated dataset of patient symptoms and their corresponding medical specialties. The dataset includes textual descriptions of symptoms and their respective labels mapped to one of the four medical departments.

The training data is collected from various medical sources, including:
- **Electronic Health Records (EHRs)**: Anonymized patient records detailing symptoms and diagnoses.
- **Medical Research Papers**: Information extracted from clinical studies.
- **Expert-Labeled Datasets**: Data manually classified by medical professionals.

Preprocessing steps include:
1. **Tokenization** using DistilBERT’s tokenizer.
2. **Stopword Removal** to eliminate non-essential words.
3. **Synonym Mapping** to standardize medical terminology.
4. **Class Balancing** to ensure equal representation of all departments.

## Model Configuration
- **Architecture:** DistilBERT for Sequence Classification
- **Hidden Dimension:** 768
- **Number of Layers:** 6
- **Attention Heads:** 12
- **Dropout:** 0.1
- **Maximum Token Length:** 512
- **Activation Function:** GELU
- **Optimizer:** AdamW
- **Batch Size:** 16
- **Epochs:** 3
- **Transformers Version:** 4.48.3

## How to Use the Model
This model can be easily loaded and used with the Hugging Face `transformers` library.

### **Installation**
```
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
```

### **Loading The Model**
```
model = AutoModelForSequenceClassification.from_pretrained("AventIQ-AI/distilbert-disease-specialist-recommendation")
tokenizer = AutoTokenizer.from_pretrained("AventIQ-AI/distilbert-disease-specialist-recommendation")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()
    return embedding

candidate_labels = ["Cardiology", "Neurology", "Orthopedics", "Dermatology"]
candidate_embeddings = np.array([encode_text(label) for label in candidate_labels])

def zero_shot_classification_batch(texts):
    text_embeddings = np.array([encode_text(text) for text in texts])
    similarities = cosine_similarity(text_embeddings, candidate_embeddings)
    ranked_results = [
        sorted(zip(candidate_labels, similarity), key=lambda x: x[1], reverse=True)
        for similarity in similarities
    ]
    return ranked_results

print(zero_shot_classification_batch(["Suffering from High Fever and body Itching"]))

# EXPECTED OUTPUT : [[('Dermatology', 0.55948263), ('Orthopedics', 0.29858905), ('Cardiology', 0.25098807), ('Neurology', 0.17517035)]]
```

## Model Files
- **model.safetensors:** The trained model weights
- **config.json:** Model configuration
- **tokenizer_config.json:** Tokenizer settings
- **special_tokens_map.json:** Special tokens used in the model
- **vocab.txt:** Vocabulary file for tokenization
