import fitz
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Initialize the RoBERTa-based question-answering pipeline
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

def extract_text_from_pdf(pdf_file_path):
    text = ""
    try:
        pdf_document = fitz.open(pdf_file_path)
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            text += page_text
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return text

if __name__ == "__main__":
    pdf_file_path = './tsne.pdf'
    pdf_text = extract_text_from_pdf(pdf_file_path)

    print("Chat with the bot (type 'exit' to end):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        answer = qa_pipeline(question=user_input, context=pdf_text)
        print("Bot:", answer['answer'])
