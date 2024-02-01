import os
import streamlit as st
import fitz
import openai
import torch
from sentence_transformers import SentenceTransformer, util
import time

# Set your OpenAI API key
openai.api_key=""

# Set up SentenceTransformer model
model=SentenceTransformer('paraphrase-MiniLM-L6-v2')
overhead_tokens=50


def extract_text_from_pdf(uploaded_file):
    try:
        # Save the uploaded PDF file temporarily
        with open("temp.pdf", "wb") as temp_pdf:
            temp_pdf.write(uploaded_file.read())

        doc=fitz.open("temp.pdf")
        text=""
        for page_num in range(doc.page_count):
            page=doc[page_num]
            text+=page.get_text()
        doc.close()
        return text
    except FileNotFoundError as e:
        st.write(f"Error: {e}")
        return None
    finally:
        # Remove the temporary file
        os.remove("temp.pdf")


def convert_text_to_vectors(text):
    return model.encode(text, convert_to_tensor=True)


def semantic_search(query_vector, pdf_vectors, pdf_texts):
    # Squeeze each tensor to make sure they are 1-dimensional
    query_vector=torch.squeeze(query_vector)
    pdf_vectors=[torch.squeeze(tensor) for tensor in pdf_vectors]

    # Convert query vector and pdf_vectors to 2D tensors
    query_vector=query_vector.unsqueeze(0)
    pdf_vectors=torch.stack(pdf_vectors)

    # Calculate cosine similarity between query vector and PDF vectors
    similarities=util.pytorch_cos_sim(query_vector, pdf_vectors)[0]

    # Sort by similarity score in descending order
    sorted_indexes=similarities.argsort(descending=True)

    # Check if any relevant information is found
    found_relevant_information=any(sim > 0 for sim in similarities)

    # Return relevant chunks and whether any relevant information was found
    relevant_chunks=[pdf_texts[i] for i in sorted_indexes]
    return relevant_chunks, found_relevant_information


# Streamlit app
st.title("PDF Text Q&A with Semantic Search")
st.write("Upload PDF files and ask questions about their content")

uploaded_files=st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    pdf_texts=[]
    pdf_vectors=[]

    for uploaded_file in uploaded_files:
        st.subheader(f"File name: {uploaded_file.name}")

        pdf_text=extract_text_from_pdf(uploaded_file)

        if pdf_text is not None:
            pdf_texts.append(pdf_text)

            # Convert text to vectors
            pdf_vector=convert_text_to_vectors(pdf_text)
            pdf_vectors.append(pdf_vector)

    # User question
    question=st.text_input("Ask a question")

    if question:
        # Convert user question to vector
        query_vector=convert_text_to_vectors(question)

        # Calculate the maximum context length dynamically
        max_context_length=4096 - len(question.split()) - overhead_tokens

        # Perform semantic search
        relevant_chunks, found_relevant_information=semantic_search(query_vector, pdf_vectors, pdf_texts)

        if found_relevant_information:
            # Construct prompt with truncated relevant chunks
            prompt="Given the information from the relevant chunks:\n\n"
            for i, chunk in enumerate(relevant_chunks):
                # Truncate each chunk to fit within the allowed context length
                truncated_chunk=chunk[:max_context_length]
                prompt+=f"Relevant Chunk {i + 1}:\n{truncated_chunk}\n\n"

            # Truncate the user query as well
            truncated_question=question[:max_context_length]
            prompt+=f"USER QUERY: {truncated_question}\n\nPlease answer the question based on the relevant information."

            try:
                # Validate that the answer is within the PDF content
                response=None
                retry_count=0

                while retry_count < 3:
                    response=openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"USER QUERY: {truncated_question}"},
                            {"role": "assistant",
                             "content": f"Given the information from the relevant chunks:\n\n{prompt}"}
                        ],
                        max_tokens=800,
                        temperature=0.5
                    )

                    if 'choices' in response and response['choices'][0]['message']['role'] == 'assistant':
                        break  # Successful response

                    retry_count+=1
                    time.sleep(20)  # Wait for 20 seconds before retrying

                if response is not None:
                    answer=response['choices'][0]['message']['content'].strip()
                    st.subheader("Answer")
                    st.write(answer)

            except Exception as e:
                st.write(f"Error: {e}")
        else:
            # No relevant information found
            st.subheader("Answer")
            st.write(f"I'm sorry, but I couldn't find any information about {question} in the provided PDFs.")
