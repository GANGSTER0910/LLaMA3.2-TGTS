# **LLaMA3.2-TGTS**  
_Text Generation and Text Summarization_

LLaMA3.2-TGTS is a state-of-the-art natural language processing (NLP) model tailored for efficient text generation and summarization tasks. Built on the recently released LLaMA 3.2 (3B) model, this project fine-tunes the model specifically for summarization, achieving highly accurate and concise results.To extend its capabilities, a custom Retrieval-Augmented Generation (RAG) model was developed from scratch using classical AI textbooks as the knowledge base, enabling effective and context-aware text generation. This innovative approach combines cutting-edge transformer architectures with curated datasets to deliver exceptional performance.
---

## **Model Details**

- **Model Name:** [`LLaMA3.2-TGTS`](https://huggingface.co/Harsh0910/LLaMA3.2-TGTS)  
- **Tasks:** 
  - Text Summarization  
  - Text Generation  

---

## **Datasets**

- **Text Summarization Dataset:**  
  [Dialog-Summarization-Dataset-Formatted](https://huggingface.co/datasets/Harsh0910/Dialog-Summarization-Dataset-Formatted)

- **Text Generation Dataset:**  
  [Session Data](https://huggingface.co/Harsh0910/my-session-data)

---

## **Installation**

To get started with LLaMA3.2-TGTS on your local system, follow the steps below:

### **1. Clone the Repository**
Clone the project repository from GitHub:  
``` markdown
git clone https://github.com/GANGSTER0910/LLaMA3.2-TGTS.git
```
Clone the repository:
``` markdown
https://github.com/GANGSTER0910/LLaMA3.2-TGTS.git
```
For running model on your own pc
```markdown
pip install -r requirements.txt
Run python Fastapi.py 
```

