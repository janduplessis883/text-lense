import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Initialize sentiment analysis pipeline
sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Initialize zero-shot classification pipeline
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Initialize emotion classification pipeline
emotion_model_name = "SamLowe/roberta-base-go_emotions"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_classifier = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, top_k=1)

def analyze_text(questions_data, total_reviews, progress_callback=None, status_callback=None):
    """
    Analyzes text reviews for sentiment, performs zero-shot classification, and detects emotions
    based on provided questions and categories.

    Args:
        questions_data (list): A list of dictionaries, each containing:
            - 'question' (str): The classification question.
            - 'reviews' (list): A list of text strings (reviews) for this question.
            - 'categories' (list): A list of category strings for the question.

    Returns:
        pandas.DataFrame: A DataFrame containing the original reviews,
                          sentiment analysis results, and zero-shot classification results
                          for each question.
    """
    all_results = []
    processed_reviews_count = 0

    for q_data in questions_data:
        question = q_data['question']
        reviews = q_data['reviews']
        categories = q_data['categories']

        if not question or not reviews or not categories:
            # Skip if essential data is missing for this question
            continue

        if status_callback:
            status_callback(f"Analyzing question: {question}")

        for review in reviews:
            if status_callback:
                status_callback(f"Analyzing review: {review[:50]}...") # Show first 50 chars of review

            # Perform sentiment analysis with truncation
            sentiment_result = sentiment_task(review, max_length=512, truncation=True)[0]
            sentiment_label = sentiment_result['label']
            sentiment_score = sentiment_result['score']

            # Perform emotion classification with truncation
            emotion_result = emotion_classifier(review, max_length=512, truncation=True)[0]
            emotion_label = emotion_result[0]['label']

            # Perform zero-shot classification with truncation
            classification_result = classifier(review, categories, multi_label=True, max_length=512, truncation=True)
            classification_labels = classification_result['labels']
            classification_scores = classification_result['scores']

            result_entry = {
                "question": question,
                "review": review,
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "emotion_label": emotion_label,
            }

            # Add classification results dynamically based on categories
            for i, label in enumerate(classification_labels):
                 result_entry[f"classification_{label}_score"] = classification_scores[i]

            # Add the top predicted label as a separate column for easier analysis
            if classification_labels:
                 result_entry[f"classification_top_label"] = classification_labels[0]
                 result_entry[f"classification_top_score"] = classification_scores[0]
            else:
                 result_entry[f"classification_top_label"] = None
                 result_entry[f"classification_top_score"] = None


            all_results.append(result_entry)

            # Update progress bar
            processed_reviews_count += 1
            if progress_callback:
                progress_callback(processed_reviews_count / total_reviews)

    print(f"Content of all_results before DataFrame conversion: {all_results}")
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    # Example usage (for testing main.py independently)
    sample_questions_data = [
        {
            "question": "Overall experience",
            "reviews": ["I had a great experience, the staff was very friendly.", "The waiting time was too long.", "Neutral feedback."],
            "categories": ["positive", "negative", "neutral"]
        },
        {
            "question": "Staff friendliness",
            "reviews": ["The staff was amazing and very helpful.", "The receptionist was rude."],
            "categories": ["friendly", "unfriendly"]
        }
    ]
    analysis_df = analyze_text(sample_questions_data)
    print(analysis_df)
