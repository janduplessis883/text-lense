FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install required packages
COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501
# Clone your Streamlit app repo
COPY . /app

# Health check (optional)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run"]

CMD ["streamlit_app.py"]
