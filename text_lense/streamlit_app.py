import streamlit as st
import pandas as pd
import plotly.express as px

# Import functions from main.py
from main import analyze_text

st.set_page_config(page_title="Text Analysis Module")

st.title("TextLense")

questions_data = [] # List to hold dictionaries: {'question': '...', 'reviews': [...], 'categories': [...]}

# Option to upload CSV or enter text
with st.sidebar:
    st.write("Upload a CSV file or enter free text reviews and configure classification questions.")
    input_method = st.radio("Choose input method:", ("Upload CSV", "Enter Text"))

    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            has_header = st.checkbox("CSV has header row", value=True)

            try:
                df = pd.read_csv(uploaded_file, header=0 if has_header else None)
                if has_header:
                    # Use column names as questions as column data as reviews
                    for col_name in df.columns:
                        question = col_name
                        reviews_list = df[col_name].dropna().tolist()
                        questions_data.append({"question": question, "reviews": reviews_list})

                else:
                    # No header, ask user to define questions and use column data as reviews
                    st.warning("CSV has no header. Please define your questions and categories below.")
                    num_columns = df.shape[1]
                    if num_columns > 0:
                        st.write(f"CSV has {num_columns} column(s). Please define a question and categories for each column.")
                        for i in range(num_columns):
                            question = st.text_input(f"Question for Column {i+1}:")
                            reviews_list = df[i].dropna().tolist()
                            questions_data.append({"question": question, "reviews": reviews_list})
                    else:
                        st.warning("CSV is empty.")


            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                questions_data = []

            # Input area for classification questions and categories (moved inside the CSV upload conditional)
            st.subheader("Classification Configuration")

            # Example structure for dynamic configuration (can be expanded)
            num_questions = st.number_input("Number of Classification Questions:", min_value=1, value=1)

            for i in range(num_questions):
                st.write(f"---")
                question = st.text_input(f"Question {i+1}:")

                # Add category prefill checkbox
                prefill_categories = st.checkbox(f"Prefill categories for Question {i+1}?", value=False)
                default_categories = ["Staff Professionalism", "Communication Effectiveness", "Appointment Availability", "Waiting Time", "Facility Cleanliness", "Patient Respect", "Treatment Quality", "Staff Empathy and Compassion", "Administrative Efficiency", "Reception Staff Interaction", "Environment and Ambiance", "Follow-up and Continuity of Care", "Accessibility and Convenience", "Patient Education and Information", "Feedback and Complaints Handling", "Test Results", "Surgery Website", "Telehealth", "Vaccinations", "Prescriptions and Medication Management", "Mental Health Support"]
                categories_str = ", ".join(default_categories) if prefill_categories else ""

                categories_input = st.text_area(f"Categories for Question {i+1} (comma-separated):", value=categories_str)
                categories = [cat.strip() for cat in categories_input.split(',') if cat.strip()]
                # Update the questions_data list with categories for the corresponding question
                if i < len(questions_data):
                    questions_data[i]["categories"] = categories
                else:
                    # This case should ideally not happen if num_questions matches the input method logic,
                    # but as a fallback, add a new entry.
                    questions_data.append({"question": question, "reviews": [], "categories": categories})


    elif input_method == "Enter Text":
        # Input area for free text reviews
        reviews_input = st.text_area("Enter Free Text Reviews (one per line):", height=200)
        if reviews_input:
            reviews_list = [review.strip() for review in reviews_input.split('\n') if review.strip()]
            # For free text, we'll need a default question or a way for the user to add one later
            # For now, let's add a placeholder question if reviews are entered
            if reviews_list:
                 questions_data.append({"question": "General Reviews", "reviews": reviews_list})
        else:
            st.warning("Please enter free text reviews.")

        # Input area for classification questions and categories (already inside this conditional)
        st.subheader("Classification Configuration")

        # Example structure for dynamic configuration (can be expanded)
        num_questions = st.number_input("Number of Classification Questions:", min_value=1, value=1)

        for i in range(num_questions):
            st.write(f"---")
            question = st.text_input(f"Question {i+1}:")

            # Add category prefill checkbox
            prefill_categories = st.checkbox(f"Prefill categories for Question {i+1}?", value=False)
            default_categories = ["Staff Professionalism", "Communication Effectiveness", "Appointment Availability", "Waiting Time", "Facility Cleanliness", "Patient Respect", "Treatment Quality", "Staff Empathy and Compassion", "Administrative Efficiency", "Reception Staff Interaction", "Environment and Ambiance", "Follow-up and Continuity of Care", "Accessibility and Convenience", "Patient Education and Information", "Feedback and Complaints Handling", "Test Results", "Surgery Website", "Telehealth", "Vaccinations", "Prescriptions and Medication Management", "Mental Health Support"]
            categories_str = ", ".join(default_categories) if prefill_categories else ""

            categories_input = st.text_area(f"Categories for Question {i+1} (comma-separated):", value=categories_str)
            categories = [cat.strip() for cat in categories_input.split(',') if cat.strip()]
            # Update the questions_data list with categories for the corresponding question
            if i < len(questions_data):
                questions_data[i]["categories"] = categories
            else:
                # This case should ideally not happen if num_questions matches the input method logic,
                # but as a fallback, add a new entry.
                questions_data.append({"question": question, "reviews": [], "categories": categories})


analyze_button = st.button("Analyze Reviews")

# Placeholder for progress bar
progress_container = st.empty()

if analyze_button:
    if questions_data:
        # Calculate total number of reviews for progress bar
        total_reviews = sum(len(q_data['reviews']) for q_data in questions_data)

        if total_reviews > 0:
            # Create a progress bar
            progress_bar = progress_container.progress(0)

            # Define a callback function to update the progress bar
            def update_progress(current_progress):
                progress_bar.progress(current_progress)

            # Use st.status to display analysis status
            with st.status("Analyzing reviews: Sentiment & Zero-shot classification...", expanded=True) as status:

                # Call analysis function with the new data structure, total reviews, progress callback, and status callback
                results_df = analyze_text(questions_data, total_reviews, update_progress, status_callback=status.write)

                # After analysis
                status.update(label="Analysis complete!", state="complete", expanded=False)
                progress_bar.progress(1.0)

                st.subheader("Analysis Results")
                # Display results
                if not results_df.empty:
                    st.dataframe(results_df)

                    # Create donut charts for sentiment distribution per question
                    try:
                        for question in results_df['question'].unique():
                            question_df = results_df[results_df['question'] == question]
                            sentiment_counts = question_df['sentiment_label'].value_counts().reset_index()
                            sentiment_counts.columns = ['sentiment_label', 'count']

                            if not sentiment_counts.empty:
                                fig = px.pie(sentiment_counts, values='count', names='sentiment_label',
                                             title=f"Sentiment Distribution for: {question}", hole=0.4)
                                st.plotly_chart(fig)

                                # Display emotion distribution as well
                                emotion_counts = question_df['emotion_label'].value_counts().reset_index()
                                emotion_counts.columns = ['emotion_label', 'count']
                                if not emotion_counts.empty:
                                    fig_emotion = px.pie(emotion_counts, values='count', names='emotion_label',
                                                          title=f"Emotion Distribution for: {question}", hole=0.4)
                                    st.plotly_chart(fig_emotion)
                                else:
                                    st.write(f"No emotion data available for question: {question}")

                            else:
                                st.write(f"No sentiment data available for question: {question}")
                    except Exception as e:
                        st.error(f"Error generating visualizations: {e}")
                else:
                    st.info("No results to display. Please check your input data and configurations.")

        else:
            st.warning("Please enter or upload reviews and configure classifications.")
