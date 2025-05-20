import streamlit as st
import pandas as pd
import plotly.express as px # Keep import in case it's needed elsewhere
import io # Import io for handling file uploads
import traceback # Import traceback for detailed error info
import altair as alt
# Import functions from main.py - Assuming main.py is in the same directory
# Import the loader functions and the analysis/visualization functions
from main import (
    analyze_text,
    visualize_results,
    load_sentiment_pipeline, # Import the loader function
    load_zero_shot_model_tokenizer, # Import the loader function
    load_emotion_model_tokenizer # Import the loader function
)

# This must be the first Streamlit command
st.set_page_config(page_title="Text Analysis Module", layout="centered") # Use wide layout

st.title(":material/reset_focus: TextLense")

# --- Initialize Models and Pipelines (AFTER set_page_config) ---
# Load the cached resources using the imported loader functions
sentiment_task = load_sentiment_pipeline()
zero_shot_model, zero_shot_tokenizer = load_zero_shot_model_tokenizer()
emotion_model, emotion_tokenizer, emotion_pipeline_tokenizer = load_emotion_model_tokenizer()

# Initialize pipelines using cached models/tokenizers
from transformers import pipeline # Import pipeline here as it's used for initialization
classifier = pipeline("zero-shot-classification", model=zero_shot_model, tokenizer=zero_shot_tokenizer)
emotion_classifier = pipeline("text-classification", model=emotion_model, tokenizer=emotion_pipeline_tokenizer, top_k=1)

# --- Sidebar for Input Method ---
with st.sidebar:
    st.header("Input Reviews")
    st.write("Upload a CSV file or enter free text reviews.")
    input_method = st.radio("Choose input method:", ("Upload CSV", "Enter Text"))

    # Store raw reviews by source in session state
    if 'raw_reviews_by_source' not in st.session_state:
        st.session_state.raw_reviews_by_source = {}

    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            has_header = st.checkbox("CSV has header row", value=True)

            try:
                # Use io.StringIO to read the uploaded file
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio, header=0 if has_header else None)

                st.session_state.raw_reviews_by_source = {} # Clear previous data

                if has_header:
                    # Use column names as potential sources
                    for col_name in df.columns:
                        # Ensure column name is a string and handle potential non-string data in column
                        reviews_list = df[col_name].dropna().astype(str).tolist()
                        if reviews_list: # Only add if there are reviews
                             st.session_state.raw_reviews_by_source[str(col_name)] = reviews_list

                else:
                    # No header, use column indices as sources
                    if df.shape[1] > 0:
                        st.write(f"CSV has {df.shape[1]} column(s). Reviews loaded from each column.")
                        for i in range(df.shape[1]):
                            # Ensure data is string and handle potential non-string data
                            reviews_list = df[i].dropna().astype(str).tolist()
                            if reviews_list: # Only add if there are reviews
                                st.session_state.raw_reviews_by_source[f"Column {i+1}"] = reviews_list
                    else:
                        st.warning("CSV is empty or could not be read.")
                        st.session_state.raw_reviews_by_source = {} # Ensure empty if CSV is empty


            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.session_state.raw_reviews_by_source = {} # Clear data on error
                st.exception(e) # Display the full exception traceback

        else:
             st.session_state.raw_reviews_by_source = {} # Clear data if no file is uploaded


    elif input_method == "Enter Text":
        # Input area for free text reviews
        reviews_input = st.text_area("Enter Free Text Reviews (one per line):", height=200)
        st.session_state.raw_reviews_by_source = {} # Clear previous data
        if reviews_input:
            reviews_list = [review.strip() for review in reviews_input.split('\n') if review.strip()]
            if reviews_list:
                 st.session_state.raw_reviews_by_source["Entered Text Reviews"] = reviews_list
        else:
            st.session_state.raw_reviews_by_source = {} # Ensure empty if no text is entered


# --- Main Content Area ---

# Display logo if no reviews are loaded
if not st.session_state.raw_reviews_by_source:
    st.write("""Please use the sidebar to upload a **CSV file** or enter **free text reviews** to start the analysis.
             :material/csv: CSV files should contain one or more columns of text reviews. It can include headers or be a simple list of reviews. :material/more_horiz: You will be given the option to select which columns to analyze.""")

    st.image('images/textlense_logo.png', width=600) # Make sure you have this image path correct
else:
    # --- Classification Configuration Section (Only show if reviews are loaded) ---
    st.header("Classification Configuration")
    st.write("Define the questions and categories for zero-shot classification and select which reviews apply to each question.")

    # Use session state to manage classification configurations dynamically
    if 'classification_configs' not in st.session_state:
        st.session_state.classification_configs = []

    def add_classification_config():
        st.session_state.classification_configs.append({'question': '', 'categories': [], 'review_sources': []})

    st.button("Add Classification Question", on_click=add_classification_config)

    # Display and edit existing configurations
    # Use enumerate with a copy to allow deletion while iterating
    # Use a copy to allow modification during iteration (deletion)
    for i, config in enumerate(st.session_state.classification_configs.copy()):
        st.write(f"---")
        st.write(f"**Classification Question {i+1}**")
        # Use unique keys for each widget based on index
        st.session_state.classification_configs[i]['question'] = st.text_input("Question:", value=config.get('question', ''), key=f'q_{i}')

        # Add category prefill checkbox
        prefill_categories = st.checkbox(f"Prefill common categories for Question {i+1}?", value=False, key=f'prefill_{i}')
        default_categories = ["Staff Professionalism", "Communication Effectiveness", "Appointment Availability", "Waiting Time", "Facility Cleanliness", "Patient Respect", "Treatment Quality", "Staff Empathy and Compassion", "Administrative Efficiency", "Reception Staff Interaction", "Environment and Ambiance", "Follow-up and Continuity of Care", "Accessibility and Convenience", "Patient Education and Information", "Feedback and Complaints Handling", "Test Results", "Surgery Website", "Telehealth", "Vaccinations", "Prescriptions and Medication Management", "Mental Health Support"]
        # Use the current categories in state if prefill is not checked, otherwise use default
        categories_str = ", ".join(default_categories) if prefill_categories else ", ".join(config.get('categories', []))

        categories_input = st.text_area("Categories (comma-separated):", value=categories_str, key=f'cat_{i}')
        st.session_state.classification_configs[i]['categories'] = [cat.strip() for cat in categories_input.split(',') if cat.strip()]

        # Allow mapping review sources to this classification question
        available_sources = list(st.session_state.raw_reviews_by_source.keys())
        if available_sources:
            selected_sources = st.multiselect(
                "Select review sources for this question:",
                available_sources,
                default=config.get('review_sources', []),
                key=f'sources_{i}'
            )
            st.session_state.classification_configs[i]['review_sources'] = selected_sources
        else:
             st.info("Upload a CSV or enter text reviews in the sidebar to select sources.")
             st.session_state.classification_configs[i]['review_sources'] = [] # Clear sources if none available

        # Button to remove this configuration
        # Use a unique key and a lambda function for the button's callback
        if st.button("Remove Question", key=f'remove_{i}'):
            # Remove the config at index i and rerun to update the UI
            del st.session_state.classification_configs[i]
            st.rerun() # Rerun the script to update the list display


    analyze_button = st.button("Analyze Reviews")

    # Placeholder for progress bar and status messages
    progress_container = st.empty()
    status_container = st.empty() # Use a separate container for status



    if analyze_button:
        questions_data_for_analysis = []
        total_reviews_for_progress = 0

        # Build the questions_data list for analyze_text based on configurations and selected sources
        for config in st.session_state.classification_configs:
            question = config.get('question', '').strip()
            categories = config.get('categories', [])
            selected_sources = config.get('review_sources', [])

            # Only include configurations that have a question, categories, and selected sources
            if question and categories and selected_sources:
                reviews_with_source_for_this_question = []
                for source in selected_sources:
                    if source in st.session_state.raw_reviews_by_source:
                        for review in st.session_state.raw_reviews_by_source[source]:
                            reviews_with_source_for_this_question.append({"review": review, "source": source})


                if reviews_with_source_for_this_question:
                     questions_data_for_analysis.append({
                         "question": question,
                         "reviews": reviews_with_source_for_this_question,
                         "categories": categories
                     })
                     total_reviews_for_progress += len(reviews_with_source_for_this_question)

        if questions_data_for_analysis:
            # Create a progress bar
            progress_bar = progress_container.progress(0)

            # Define a callback function to update the progress bar
            def update_progress(current_progress):
                progress_bar.progress(current_progress)

            # Use st.status to display analysis status
            # status_container is used here
            results_df = pd.DataFrame() # Initialize results_df outside the try block

            try:
                with st.status("Analyzing reviews: Sentiment, Emotion & Zero-shot classification...", expanded=True) as status:

                    # Define a callback function to update the status message
                    def update_status(message):
                        status.write(message) # Write message inside the status expander

                    # Call analysis function with the prepared data and initialized pipelines
                    analysis_results_list = analyze_text(
                        questions_data_for_analysis,
                        total_reviews_for_progress,
                        sentiment_task, # Pass the initialized pipeline
                        classifier, # Pass the initialized pipeline
                        emotion_classifier, # Pass the initialized pipeline
                        progress_callback=update_progress,
                        status_callback=update_status
                    )

                    # --- Specific Error Handling and Debugging for DataFrame Creation and Display ---
                    results_df = pd.DataFrame()
                    try:
                        results_df = pd.DataFrame(analysis_results_list)
                        print("\n--- Debugging results_df ---")
                        print(f"DataFrame created successfully. Shape: {results_df.shape}") # Print shape
                        print(f"Columns: {results_df.columns.tolist()}") # Print columns
                        print("Head of DataFrame:")
                        print(results_df.head()) # Print head of DataFrame to terminal
                        print("--- End Debugging results_df ---\n")

                    except Exception as df_error:
                        st.error(f"Error creating DataFrame: {df_error}")
                        st.exception(df_error)
                        results_df = pd.DataFrame() # Ensure results_df is empty on error
                    # --- End Specific Error Handling and Debugging ---

                    # After analysis
                    status.update(label="Analysis complete!", state="complete", expanded=False)
                    progress_bar.progress(1.0) # Ensure progress bar reaches 100%

                # Clear progress bar and status container after analysis is complete and status is closed
                # Add a small delay maybe? Sometimes clearing too fast can affect rendering.
                # import time
                # time.sleep(0.1) # Optional: Add a tiny delay
                progress_container.empty()
                status_container.empty()

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e) # Display the full exception traceback
                # You might want to clear the progress and status containers here too on error
                progress_container.empty()
                status_container.empty()

            # Display results outside the st.status block
            st.subheader("Analysis Results")

            # Visualize the results
            if not results_df.empty:
                visualize_results(results_df)

                # Display results DataFrame below visualizations
                st.write("### Raw Analysis Data")
                st.dataframe(results_df, width=2000) # Set a large width
            else:
                st.info("No results to display. Please check your input data and configurations.")

        else:
            st.warning("Please configure at least one classification question with categories and select review sources, and ensure reviews are loaded.")
