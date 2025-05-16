import streamlit as st
import pandas as pd
import plotly.express as px # Keep import in case it's needed elsewhere, though not for display
import io # Import io for handling file uploads
import traceback # Import traceback for detailed error info

# Import functions from main.py
from main import analyze_text

st.set_page_config(page_title="Text Analysis Module", layout="wide") # Use wide layout

st.title(":material/reset_focus: TextLense")

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

    st.image('images/textlense_logo.png')
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
    for i, config in enumerate(st.session_state.classification_configs):
        st.write(f"---")
        st.write(f"**Classification Question {i+1}**")
        # Use unique keys for each widget based on index
        config['question'] = st.text_input("Question:", value=config.get('question', ''), key=f'q_{i}')

        # Add category prefill checkbox
        prefill_categories = st.checkbox(f"Prefill common categories for Question {i+1}?", value=False, key=f'prefill_{i}')
        default_categories = ["Staff Professionalism", "Communication Effectiveness", "Appointment Availability", "Waiting Time", "Facility Cleanliness", "Patient Respect", "Treatment Quality", "Staff Empathy and Compassion", "Administrative Efficiency", "Reception Staff Interaction", "Environment and Ambiance", "Follow-up and Continuity of Care", "Accessibility and Convenience", "Patient Education and Information", "Feedback and Complaints Handling", "Test Results", "Surgery Website", "Telehealth", "Vaccinations", "Prescriptions and Medication Management", "Mental Health Support"]
        # Use the current categories in state if prefill is not checked
        categories_str = ", ".join(default_categories) if prefill_categories else ", ".join(config.get('categories', []))

        categories_input = st.text_area("Categories (comma-separated):", value=categories_str, key=f'cat_{i}')
        config['categories'] = [cat.strip() for cat in categories_input.split(',') if cat.strip()]

        # Allow mapping review sources to this classification question
        available_sources = list(st.session_state.raw_reviews_by_source.keys())
        if available_sources:
            selected_sources = st.multiselect(
                "Select review sources for this question:",
                available_sources,
                default=config.get('review_sources', []),
                key=f'sources_{i}'
            )
            config['review_sources'] = selected_sources
        else:
             st.info("Upload a CSV or enter text reviews in the sidebar to select sources.")
             config['review_sources'] = [] # Clear sources if none available

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

    # --- Simple Test DataFrame Display (Moved inside the else block) ---
    st.write("---")
    st.write("### Test DataFrame Display (Visible when reviews are loaded)")
    try:
        test_df = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
        st.dataframe(test_df)
        st.write("If you see the DataFrame above, Streamlit's basic DataFrame display is working.")
    except Exception as e:
        st.error(f"Error displaying test DataFrame: {e}")
        st.exception(e)
    st.write("---")
    # --- End Test DataFrame Display ---


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
                reviews_for_this_question = []
                for source in selected_sources:
                    if source in st.session_state.raw_reviews_by_source:
                        reviews_for_this_question.extend(st.session_state.raw_reviews_by_source[source])

                if reviews_for_this_question:
                     questions_data_for_analysis.append({
                         "question": question,
                         "reviews": reviews_for_this_question,
                         "categories": categories
                     })
                     total_reviews_for_progress += len(reviews_for_this_question)

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

                    # Call analysis function with the prepared data
                    analysis_results_list = analyze_text(
                        questions_data_for_analysis,
                        total_reviews_for_progress,
                        progress_callback=update_progress,
                        status_callback=update_status
                    )

                    # --- Specific Error Handling and Debugging for DataFrame Creation and Display ---
                    try:
                        results_df = pd.DataFrame(analysis_results_list)
                        print("\n--- Debugging results_df ---")
                        print(f"DataFrame created successfully. Shape: {results_df.shape}") # Print shape
                        print(f"Columns: {results_df.columns.tolist()}") # Print columns
                        print("Head of DataFrame:")
                        print(results_df.head()) # Print head of DataFrame to terminal
                        print("--- End Debugging results_df ---\n")

                        st.subheader("Analysis Results")
                        # Display results - Try setting width explicitly
                        if not results_df.empty:
                            st.dataframe(results_df, width=2000) # Set a large width
                            # If the above doesn't work, try displaying a subset of columns
                            # st.write("Attempting to display a subset of columns:")
                            # subset_cols = ['question', 'review', 'sentiment_label', 'emotion_label', 'classification_top_label']
                            # display_cols = [col for col in subset_cols if col in results_df.columns]
                            # if display_cols:
                            #     st.dataframe(results_df[display_cols])
                            # else:
                            #      st.write("Could not find key columns for subset display.")

                        else:
                            st.info("No results to display. Please check your input data and configurations.")

                    except Exception as df_error:
                        st.error(f"Error creating or displaying DataFrame: {df_error}")
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


        else:
            st.warning("Please configure at least one classification question with categories and select review sources, and ensure reviews are loaded.")
