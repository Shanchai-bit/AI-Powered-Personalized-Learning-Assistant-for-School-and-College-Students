from transformers import pipeline

def summarize_text(text):
    # Initialize summarization pipeline with pretrained model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    try:
        # Generate summary with length constraints
        summary = summarizer(text, max_length=150, min_length=25, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        # Return error message on failure
        return f"Error during summarization: {e}"
