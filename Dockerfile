FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install required packages
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone your Streamlit app repo
RUN git clone https://github.com/janduplessis883/text-lense.git

# Set working directory inside the cloned repo
WORKDIR /app/text-lense

# Optional cache fix for Hugging Face models
RUN mkdir -p /app/hf_cache /app/hf_home /app/.cache
ENV TRANSFORMERS_CACHE=/app/hf_cache \
    HF_HOME=/app/hf_home \
    XDG_CACHE_HOME=/app/.cache

# Copy secrets if needed (or mount as volume)
COPY .streamlit/ .streamlit/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Health check (optional)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
