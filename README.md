![TextLense Logo](images/textlense_logo.png)

# 🔍 TextLense — Interactive Text Analysis App

**TextLense** is a 📊 Streamlit-powered app for exploring and analyzing text data.

Upload text reviews from a CSV or enter text directly. Instantly uncover insights with:

- 💬 **Sentiment Analysis**
  Detects if the sentiment is **positive**, **negative**, or **neutral**.

- 🧠 **Zero-Shot Classification**
  Create your own categories and classify text without needing labeled data.

- 😄 **Emotion Detection**
  Identifies the **dominant emotion** in the text (e.g. joy, anger, sadness).

Interactive charts and raw data views help you dive deep into your dataset.

---

## ⚙️ Installation & Setup

### 🐳 1. Install Docker

Download [Docker Desktop](https://docs.docker.com/get-started/get-docker/) for Windows, macOS, or Linux.

### 📥 2. Pull the Docker Image

Open a terminal and run:

```bash
docker pull janduplessis883/text-lense
```

### 🚀 3. Run the App

Start the container:

```bash
docker run -p 8501:8501 janduplessis883/text-lense
```

You’ll see something like this in your terminal:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.17.0.5:8501
  External URL: http://185.108.105.142:8501
```

### 🌐 4. Open in Browser

Go to [http://localhost:8501](http://localhost:8501) to start using TextLense.
