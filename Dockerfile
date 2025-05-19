FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    gnupg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC drivers
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl https://packages.microsoft.com/config/debian/12/prod.list | tee /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Optional: clone your repo (remove if not needed)
RUN git clone https://github.com/janduplessis883/text-lense.git

# Create cache directories to avoid permission errors
RUN mkdir -p /app/hf_cache /app/hf_home /app/.cache

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/hf_cache \
    HF_HOME=/app/hf_home \
    XDG_CACHE_HOME=/app/.cache

# Copy your Streamlit app and .streamlit config
COPY . ./
COPY .streamlit/ .streamlit/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start Streamlit app
ENTRYPOINT ["streamlit", "run", "index.py", "--server.port=8501", "--server.address=0.0.0.0"]
