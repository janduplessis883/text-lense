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
    load_emotion_model_tokenizer, # Import the loader function
    load_segmentation_models # Import the segmentation loader function
)

# This must be the first Streamlit command
st.set_page_config(page_title="Text Analysis Module", layout="centered") # Use wide layout

st.title(":material/reset_focus: TextLense")
st.sidebar.header("Configuration")
zero_shot_model = st.sidebar.selectbox("Select a zero-shot classification model:", options=['MoritzLaurer/deberta-v3-large-zeroshot-v2.0', 'facebook/bart-large-mnli'], index=1)
default_selector = st.sidebar.selectbox("Select default Categories Set:", options=['Classic Set', 'Primary Thematic Set', 'Primary Thematic Set Modified v6', 'Blank'], index=1)

# Add segmentation toggle in sidebar
st.sidebar.divider()
st.sidebar.subheader("Advanced Options")
use_segmentation = st.sidebar.toggle(
    "Enable Review Segmentation",
    value=False,
    help="Segment long reviews into smaller parts for more accurate aspect-based analysis. Uses SentenceTransformer for initial segmentation, and LLM if more than 6 segments are detected."
)
llm_model = st.sidebar.text_input(
    "LLM Model for Segmentation",
    value="gpt-oss:20b",
    help="Ollama model name for LLM-based segmentation (used when >6 segments detected)",
    disabled=not use_segmentation
)

# --- Initialize Models and Pipelines (AFTER set_page_config) ---
# Load the cached resources using the imported loader functions
sentiment_task = load_sentiment_pipeline()
zero_shot_model, zero_shot_tokenizer = load_zero_shot_model_tokenizer(zero_shot_model)
emotion_model, emotion_tokenizer, emotion_pipeline_tokenizer = load_emotion_model_tokenizer()

# Load segmentation models if enabled
embedder = None
nlp = None
if use_segmentation:
    with st.spinner("Loading segmentation models..."):
        embedder, nlp = load_segmentation_models()
        if embedder is not None:
            st.sidebar.success("✓ Segmentation models loaded")
        else:
            st.sidebar.error("✗ Failed to load segmentation models")
            use_segmentation = False  # Disable if models failed to load

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

    # Store human labels in session state
    if 'human_labels' not in st.session_state:
        st.session_state.human_labels = {}

    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            has_header = st.checkbox("CSV has header row", value=True)

            # Checkbox to indicate if CSV contains human-labelled data
            has_human_labels = st.checkbox(
                "CSV includes 'human_labelled' column",
                value=False,
                help="Check this if your CSV contains a 'human_labelled' column for comparison with AI classifications"
            )

            try:
                # Use io.StringIO to read the uploaded file
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                df = pd.read_csv(stringio, header=0 if has_header else None)

                temp_human_labels = [] # Initialize here, will be populated after filtering

                # Replace empty strings with NaN for consistent handling, but do not drop rows based on 'any' NA.
                # We will filter based on review content length instead.
                df = df.replace('', pd.NA)

                # Initialize a Series to track which rows to keep.
                # A row is kept if AT LEAST ONE review column has > 5 words.
                rows_to_keep_by_review_content = pd.Series([False] * len(df), index=df.index)
                review_columns_to_check = []

                if has_header:
                    for col_name in df.columns:
                        if col_name == 'human_labelled':
                            continue
                        review_columns_to_check.append(col_name)
                else:
                    # If no header, assume all columns are potential review columns for filtering purposes.
                    # This is a simplification; a more robust solution might ask the user to specify review columns.
                    review_columns_to_check = df.columns.tolist()

                if not review_columns_to_check:
                    st.warning("No review columns identified for filtering by word count. All rows will be kept.")
                    rows_to_keep_by_review_content = pd.Series([True] * len(df), index=df.index)
                else:
                    for col in review_columns_to_check:
                        # Count words in each review. Treat NaN/empty strings as 0 words.
                        # The .fillna('') ensures .astype(str) doesn't convert NaN to 'nan' string.
                        word_counts = df[col].fillna('').astype(str).apply(lambda x: len([w for w in x.strip().split(' ') if w]))
                        # A row is kept if it meets the word count for THIS column OR it already met it for another column
                        rows_to_keep_by_review_content = rows_to_keep_by_review_content | (word_counts > 5)

                df = df[rows_to_keep_by_review_content]
                st.info(f"ℹ️ After filtering short reviews: {len(df)} rows remain")

                # Now, extract human labels from the *filtered* dataframe if the column exists
                if has_human_labels and has_header and 'human_labelled' in df.columns:
                    temp_human_labels = df['human_labelled'].astype(str).tolist()
                    st.success(f"✓ Found 'human_labelled' column with {len(temp_human_labels)} labels (after filtering)")

                st.session_state.raw_reviews_by_source = {} # Clear previous data
                st.session_state.human_labels = {} # Clear previous human labels

                # Collect all reviews from the filtered dataframe
                temp_reviews_by_source = {}

                if has_header:
                    # Use column names as potential sources
                    for col_name in df.columns:
                        # Skip the human_labelled column when processing reviews
                        if col_name == 'human_labelled':
                            continue

                        # Extract reviews directly from the already-filtered dataframe
                        reviews_list = df[col_name].astype(str).tolist()
                        reviews_list = [r.strip() for r in reviews_list if r.strip()]

                        if reviews_list: # Only add if there are reviews
                             temp_reviews_by_source[str(col_name)] = reviews_list

                else:
                    # No header, use column indices as sources
                    if df.shape[1] > 0:
                        st.write(f"CSV has {df.shape[1]} column(s). Reviews loaded from each column.")
                        for i in range(df.shape[1]):
                            # Ensure data is string and handle potential non-string data
                            reviews_list = df[i].dropna().astype(str).tolist()
                            reviews_list = [r for r in reviews_list if r.strip()]
                            if reviews_list: # Only add if there are reviews
                                temp_reviews_by_source[f"Column {i+1}"] = reviews_list
                    else:
                        st.warning("CSV is empty or could not be read.")
                        st.session_state.raw_reviews_by_source = {} # Ensure empty if CSV is empty

                # Apply slicing if we have reviews
                if temp_reviews_by_source:
                    # Get the maximum length of reviews across all sources
                    max_length = max(len(reviews) for reviews in temp_reviews_by_source.values())

                    st.write("---")
                    st.write("### Slice Reviews")
                    st.write(f"Total **reviews available**: {max_length}")

                    # Number inputs for slicing
                    start_index = st.number_input(
                        "Start Index:",
                        min_value=0,
                        max_value=max(0, max_length - 1),
                        value=0,
                        help="Starting index for slicing reviews (inclusive)"
                    )
                    end_index = st.number_input(
                        "End Index:",
                        min_value=start_index + 1,
                        max_value=max_length,
                        value=max_length,
                        help="Ending index for slicing reviews (exclusive)"
                    )

                    # Apply slicing to all review sources
                    for source, reviews in temp_reviews_by_source.items():
                        sliced_reviews = reviews[start_index:end_index]
                        if sliced_reviews:
                            st.session_state.raw_reviews_by_source[source] = sliced_reviews

                    # Apply slicing to human labels if they exist
                    if temp_human_labels:
                        sliced_labels = temp_human_labels[start_index:end_index]
                        st.session_state.human_labels['sliced'] = sliced_labels
                        st.info(f"Using reviews from index {start_index} to {end_index} ({end_index - start_index} reviews per source) with {len(sliced_labels)} human labels")
                    else:
                        st.info(f"Using reviews from index {start_index} to {end_index} ({end_index - start_index} reviews per source)")


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
            # Filter out reviews with less than 6 words
            filtered_reviews_list = []
            for r in reviews_list:
                testlist = [word for word in r.strip().split(' ') if word]
                if len(testlist) > 5:
                    filtered_reviews_list.append(' '.join(testlist))
            reviews_list = filtered_reviews_list
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
    st.subheader("Classification Configuration")
    st.caption("Define the questions and categories for zero-shot classification and select which reviews apply to each question.")

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

        default_categories1 = ["Staff Professionalism", "Communication Effectiveness", "Appointment Availability", "Waiting Time", "Facility Cleanliness", "Patient Respect", "Treatment Quality", "Staff Empathy and Compassion", "Administrative Efficiency", "Reception Staff Interaction", "Environment and Ambiance", "Follow-up and Continuity of Care", "Accessibility and Convenience", "Patient Education and Information", "Feedback and Complaints Handling", "Test Results", "Surgery Website", "Telehealth", "Vaccinations", "Prescriptions and Medication Management", "Mental Health Support", "Unclassifiable"]
        default_categories2 = ['Access & Availability', 'Provision of Information', 'Privacy & Confidentiality', 'Continuity of Care', 'Clinical Staff Interpersonal Skills', 'Administrative Staff Interpersonal Skills', 'Service Delivery Process', 'Clinical Quality & Safety', 'Unclassifiable']
        default_categories3 = ["Appointment Availability & Access",
                                "Online Services",
                                "Admin Processes",
                                "Facilities & Cleanliness",
                                "Prescription & Medication Management",
                                "Clinical Treatment Quality",
                                "Test Results & Follow-up",
                                "Health Promotion",
                                "Nursing & HCA Services",
                                "Reception Staff Interaction",
                                "Clinical Staff Professionalism & Attitude",
                                "Continuity of Care",
                                "Privacy & Confidentiality",
                                "Feedback & Complaints Process",
                                "Overall Experience"
                                ]
        # Initialize categories_input
        categories_input = ""

        if default_selector == 'Classic Set':
            categories_input = st.text_area(
                    "Categories (comma-separated):",
                    value="Staff Professionalism,Communication Effectiveness,Appointment Availability,Waiting Time,Facility Cleanliness,Patient Respect,Treatment Quality,Staff Empathy and Compassion,Administrative Efficiency,Reception Staff Interaction,Environment and Ambiance,Follow-up and Continuity of Care,Accessibility and Convenience,Patient Education and Information,Feedback and Complaints Handling,Test Results,Surgery Website,Telehealth,Vaccinations,Prescriptions and Medication Management,Mental Health Support,Unclassifiable",
                    key=f'cat_{i}'
            )
        elif default_selector == 'Primary Thematic Set':
            categories_input = st.text_area(
                    "Categories (comma-separated):",
                    value="Access & Availability,Provision of Information,Privacy & Confidentiality,Continuity of Care,Clinical Staff Interpersonal Skills,Administrative Staff Interpersonal Skills,Service Delivery Process,Clinical Quality & Safety,Unclassifiable",
                    key=f'cat_{i}'
            )
        elif default_selector == 'Primary Thematic Set Modified v6':
            categories_input = st.text_area(
                    "Categories (comma-separated):",
                    value="Appointment Availability & Access,Online Services,Admin Processes,Facilities & Cleanliness,Prescription & Medication Management,Clinical Treatment Quality,Test Results & Follow-up,Health Promotion,Nursing & HCA Services,Reception Staff Interaction,Clinical Staff Professionalism & Attitude,Continuity of Care,Privacy & Confidentiality,Feedback & Complaints Process,Overall Experience",
                    key=f'cat_{i}'
            )
        elif default_selector == 'Blank':
            categories_input = st.text_area(
                    "Categories (comma-separated):",
                    value="",
                    key=f'cat_{i}'
            )

        # Process and store categories
        if categories_input.strip():
            st.session_state.classification_configs[i]['categories'] = [cat.strip() for cat in categories_input.split(',') if cat.strip()]
        else:
            st.session_state.classification_configs[i]['categories'] = []


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


    analyze_button = st.button("Analyze Reviews", type='primary')

    # Placeholder for progress bar and status messages
    progress_container = st.empty()
    status_container = st.empty() # Use a separate container for status

    # Initialize results_df in session state to persist across reruns
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

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
            # results_df = pd.DataFrame() # No longer needed as it's in session state

            try:
                with st.status("Analyzing reviews: Sentiment Analysis, Emotion Detection & Zero-shot classification...", expanded=True) as status:

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
                        status_callback=update_status,
                        use_segmentation=use_segmentation,
                        embedder=embedder,
                        llm_model=llm_model
                    )

                    # --- Specific Error Handling and Debugging for DataFrame Creation and Display ---
                    # results_df = pd.DataFrame() # No longer needed as it's in session state
                    try:
                        st.session_state.results_df = pd.DataFrame(analysis_results_list)

                        # Debug: Check session state
                        print("\n--- Debugging Human Labels ---")
                        print(f"Session state human_labels keys: {st.session_state.human_labels.keys()}")
                        if 'sliced' in st.session_state.human_labels:
                            print(f"Sliced labels length: {len(st.session_state.human_labels['sliced'])}")
                            print(f"First few labels: {st.session_state.human_labels['sliced'][:5]}")
                        print(f"Results DataFrame length: {len(st.session_state.results_df)}")
                        print("--- End Debugging Human Labels ---\n")

                        # Merge human labels if they exist
                        if 'sliced' in st.session_state.human_labels and st.session_state.human_labels['sliced']:
                            human_labels = st.session_state.human_labels['sliced']
                            # Ensure the lengths match
                            if len(human_labels) == len(st.session_state.results_df):
                                st.session_state.results_df['HUMAN LABELS'] = human_labels
                                status.write(f"✓ Added 'HUMAN LABELS' column with {len(human_labels)} human labels for comparison")
                            else:
                                status.write(f"⚠️ Human labels count ({len(human_labels)}) doesn't match results count ({len(st.session_state.results_df)}). Skipping merge.")
                        else:
                            status.write("ℹ️ No human labels found in session state")

                        print("\n--- Debugging results_df ---")
                        print(f"DataFrame created successfully. Shape: {st.session_state.results_df.shape}") # Print shape
                        print(f"Columns: {st.session_state.results_df.columns.tolist()}") # Print columns
                        print("Head of DataFrame:")
                        print(st.session_state.results_df.head()) # Print head of DataFrame to terminal
                        print("--- End Debugging results_df ---")

                    except Exception as df_error:
                        st.error(f"Error creating DataFrame: {df_error}")
                        st.exception(df_error)
                        st.session_state.results_df = pd.DataFrame() # Ensure results_df is empty on error
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
        if not st.session_state.results_df.empty:
            visualize_results(st.session_state.results_df)

            # --- Calculate and Display Accuracy Metrics (New Section) ---
            if 'HUMAN LABELS' in st.session_state.results_df.columns:
                st.write("### Accuracy Metrics (Human vs. AI Labels)")
                accuracy_data = []

                # Get unique categories from the AI classification
                unique_ai_categories = st.session_state.results_df['classification_top_label'].unique()

                for category in unique_ai_categories:
                    # Filter for rows where AI classified as this category
                    category_df = st.session_state.results_df[st.session_state.results_df['classification_top_label'] == category]

                    if not category_df.empty:
                        # Count how many of these AI classifications match the human label
                        correct_predictions = (category_df['HUMAN LABELS'] == category_df['classification_top_label']).sum()
                        total_predictions = len(category_df)

                        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
                        accuracy_data.append({'Category': category, 'Accuracy (%)': f"{accuracy:.2f}%", 'Total AI Predictions': total_predictions, 'Correct AI Predictions': correct_predictions})

                if accuracy_data:
                    accuracy_df = pd.DataFrame(accuracy_data)
                    st.dataframe(accuracy_df)
                else:
                    st.info("No accuracy metrics to display. Ensure human labels are present and categories are classified.")
            # --- End Accuracy Metrics Section ---

            # --- Sentiment Heatmap Section (New) ---
            st.write("### Sentiment Analysis Heatmap by Category")
            if not st.session_state.results_df.empty and 'classification_top_label' in st.session_state.results_df.columns and all(col in st.session_state.results_df.columns for col in ['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']):
                # Group by classification_top_label and calculate mean sentiment scores
                sentiment_by_category = st.session_state.results_df.groupby('classification_top_label')[
                    ['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']
                ].mean().reset_index()

                # Rename columns for better display in heatmap
                sentiment_by_category.rename(columns={
                    'classification_top_label': 'Category',
                    'sentiment_positive': 'Mean Positive Sentiment',
                    'sentiment_neutral': 'Mean Neutral Sentiment',
                    'sentiment_negative': 'Mean Negative Sentiment'
                }, inplace=True)

                if not sentiment_by_category.empty:
                    # Melt the DataFrame for Plotly Express heatmap
                    melted_sentiment = sentiment_by_category.melt(
                        id_vars='Category',
                        var_name='Sentiment Type',
                        value_name='Mean Score'
                    )

                    # Pivot the melted DataFrame to get a matrix for imshow
                    heatmap_data = melted_sentiment.pivot(index='Category', columns='Sentiment Type', values='Mean Score')

                    # Ensure the order of sentiment types for consistent display
                    sentiment_order = ['Mean Positive Sentiment', 'Mean Neutral Sentiment', 'Mean Negative Sentiment']
                    heatmap_data = heatmap_data[sentiment_order]

                    # Create the heatmap using Plotly Express imshow
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x="Sentiment Type", y="Category", color="Mean Score"),
                        x=heatmap_data.columns.tolist(),
                        y=heatmap_data.index.tolist(),
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title='Mean Sentiment Score by Category',
                        text_auto=True # Display values on heatmap
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No categories found for sentiment heatmap.")
            else:
                st.info("No data available to generate sentiment heatmap by category. Ensure reviews are classified and sentiment scores are present.")
            # --- End Sentiment Heatmap Section ---

            # Display results DataFrame below visualizations
            st.write("### Raw Analysis Data")
            st.dataframe(st.session_state.results_df, width=2000) # Set a large width

            # Add download button for CSV export
            csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=":material/download: Download Results as CSV",
                data=csv,
                file_name='text_analysis_results.csv',
                mime='text/csv',
                help="Download the complete analysis results as a CSV file"
            )
        else:
            st.info("No results to display. Please check your input data and configurations.")

    else:
        st.warning("Please configure at least one classification question with categories and select review sources, and ensure reviews are loaded.")
