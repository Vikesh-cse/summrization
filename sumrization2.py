import streamlit as st 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.title("📑 Text Summariser ")

model = ChatGroq(
    model='llama-3.1-8b-instant',
    temperature=0.3
)

num_paragraph = st.number_input(
    "Enter the number of paragraph for summary (optional)",
    min_value=1,
    max_value=15,
    value= 6
)

upload_file = st.file_uploader("Upload your pdf here ", type='pdf')

if upload_file:
    
    reader = PdfReader(upload_file)
    
    text = ""
    
    for pages in reader.pages:
        content = pages.extract_text()
        
        if content:
            text += content
    
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    
    chunks = spliter.split_text(text)
    
    if st.button("Generate Summary"):
        
        combined_text = ""

        for chunk in chunks[:5]:
            combined_text += chunk + "\n"

        prompt = f"""
You are an expert university professor and subject matter expert.

Your task is to analyze the following academic content and explain it in a clear, deep, and easy-to-understand manner for university students.

Instructions:
1. Write the explanation in exactly {num_paragraph} well-structured paragraphs.
2. Use simple language so beginners can understand.
3. Focus on important academic concepts.
4. If useful, include real-world examples or analogies.

Content to analyze:
{combined_text} 

Generate the output in the following structured format:

1. Bullet Point Summary
- Provide concise bullet points summarizing the main ideas.

2. Detailed Explanation
- Explain the topic clearly in {num_paragraph} paragraphs.

3. Key Concepts
- List the most important concepts students must remember.

2 Marks Questions:
- Generate 5 short conceptual questions.

7 Marks Questions:
- Generate 5 descriptive exam questions suitable for long answers.

Make the output clean, structured, and easy for students to study from.
"""

        response = model.invoke(prompt)

        st.subheader("Summary")
        st.write(response.content)