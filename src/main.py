import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
import plotly.express as px
import altair as alt

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

        for review_data in reviews:
            review = review_data['review']
            source = review_data['source'] # Extract source

            if status_callback:
                status_callback(f"Analyzing review from '{source}': {review[:50]}...") # Show first 50 chars of review and source

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
                "source": source, # Include source in result_entry
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

def visualize_results(results_df):
    """
    Visualizes the analysis results using Streamlit.

    Args:
        results_df (pandas.DataFrame): DataFrame containing the analysis results.
    """
    st.subheader("Analysis Results Visualization")

    if results_df.empty:
        st.info("No results to visualize.")
        return

    # Sentiment Distribution by Source (Donut Charts)
    st.write("### Sentiment Distribution by Source")
    if 'source' in results_df.columns:
        for source in results_df['source'].unique():
            st.write(f"#### Source: {source}")
            source_df = results_df[results_df['source'] == source]
            sentiment_counts = source_df['sentiment_label'].value_counts().reset_index()
            sentiment_counts.columns = ['sentiment_label', 'count']

            # Define custom colors
            color_map = {'positive': '#7b94a6', 'negative': '#a25450', 'neutral': '#edeadc'}

            fig = px.pie(sentiment_counts, values='count', names='sentiment_label',
                         title=f'Sentiment Distribution for {source}',
                         hole=0.5, # Creates the donut shape
                         color='sentiment_label',
                         color_discrete_map=color_map) # Apply custom colors
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Source information not available for sentiment visualization.")


    # Emotion Distribution by Source (Bar Charts)
    st.write("### Emotion Distribution by Source")
    if 'source' in results_df.columns:
        for source in results_df['source'].unique():
            st.write(f"#### Source: {source}")
            source_df = results_df[results_df['source'] == source]
            emotion_counts = source_df['emotion_label'].value_counts().reset_index()
            emotion_counts.columns = ['emotion_label', 'count']
            chart = alt.Chart(emotion_counts).mark_bar(color='#ca8f56').encode(
                x=alt.X('count:Q', title='Count'),
                y=alt.Y('emotion_label:N', title='Emotion')
            ).properties(
                title=f'Emotion Distribution for {source}'
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Source information not available for emotion visualization.")


    # Top Classification Label Distribution per Question (Bar Charts - unchanged)
    st.write("### Top Classification Label Distribution per Question")
    for question in results_df['question'].unique():
        st.write(f"#### Question: {question}")
        question_df = results_df[results_df['question'] == question]
        if 'classification_top_label' in question_df.columns:
            classification_counts = question_df['classification_top_label'].value_counts().reset_index()
            classification_counts.columns = ['classification_top_label', 'count']
            chart = alt.Chart(classification_counts).mark_bar(color='#1d2d3d').encode(
                x=alt.X('count:Q', title='Count'),
                y=alt.Y('classification_top_label:N', title='Classification Label')
            ).properties(
                title=f'Top Classification Label Distribution for {question}'
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No classification results for this question.")


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
    st.dataframe(analysis_df)
