I built a simple RAG chatbot where user will ask questions and the the RAG will then index from the files provided and provide the final answer. everything happening here is using local models.
used local embedding model, local chat model, local reranker model. The retreival is happening in hybrid way. It is taking semantic context as well as BM25. 

now for verification, I asked the following question: " Which sector is the backbone of the backbone of India's national economy?"

The answer to the question provided by the chat model without providing the context is following:

<img width="832" height="95" alt="Screenshot 2026-09-01 223410" src="https://github.com/user-attachments/assets/b5690bbf-7d79-4bcc-acc7-c83c5f850f3c" />



Then when I asked the RAg model the same question it answered the following:

<img width="674" height="86" alt="Screenshot 2026-09-01 223424" src="https://github.com/user-attachments/assets/7e184475-673b-45de-8168-51dcc371e356" />



Now i will give what was written in the context (actual):

<img width="725" height="47" alt="image" src="https://github.com/user-attachments/assets/0420ba30-7fb3-47f5-9a9e-e5ce1feddc2f" />




