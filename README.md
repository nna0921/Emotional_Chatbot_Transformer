# 💬 MoodBud: Empathetic Transformer Chatbot 

**MoodBud** is an advanced conversational AI built on a **Transformer-based sequence-to-sequence model** designed to generate **emotionally aware and empathetic responses**. Leveraging the rich context of the Empathetic Dialogues dataset, this project provides an interactive Streamlit interface for exploring the model's capabilities.

The core model is initialized with a specific **Emotion** and **Situation** to guide the conversation's empathetic tone.

---

## Features

* **Empathetic Responses:** Generates responses conditioned on a specified emotional state and situation.
* **Transformer Architecture:** Utilizes a custom PyTorch implementation of the original Transformer (*Attention Is All You Need*) model.
* **Decoding Strategies:** Supports both **Greedy Decoding** and **Beam Search** for exploring different response generation techniques.
* **Persistent Chat History:** Maintains conversation context within the Streamlit session.
* **Automatic Data & Model Download:** Automatically downloads the required CSV dataset and the pre-trained PyTorch model from Google Drive upon first run.
* **Tokenization & Vocabulary:** Custom tokenization, vocabulary creation, and text normalization tailored for the dialogue data.

---

## Setup and Installation

### Prerequisites

You need **Python 3.8+** installed on your system.

### Steps

1.  **Save the Code:**
    Ensure the provided Python code is saved as a file named `app.py`.

2.  **Install Required Libraries:**
    Create a `requirements.txt` file with the following dependencies and install them:

    ```bash
    # requirements.txt
    streamlit
    pandas
    scikit-learn
    numpy
    torch
    gdown
    ```

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    Launch the Streamlit app from your terminal:

    ```bash
    streamlit run app.py
    ```

    The application will automatically open in your web browser, typically at `http://localhost:8501`.

**Note:** The first time you run the application, it will automatically download a $\approx 3.7$MB CSV file (`emotion-emotion_69k.csv`) and a $\approx 8$MB pre-trained model file (`best_transformer_chatbot (1).pt`) from Google Drive.

---

## 💡 How to Use the App

The application uses a **sidebar** for configuration and the **main panel** for the chat interface.

### 1. Configure the Context (Sidebar)

Before starting the conversation, use the sidebar to set the context for the agent:

* **Emotion:** Select the target emotion the agent should be addressing (e.g., *Anger*, *Sadness*, *Frustration*).
* **Decoding:** Choose the generation strategy:
    * `Greedy`: Selects the token with the highest probability at each step.
    * `Beam Search`: Explores multiple high-probability sequences for a more coherent result.
* **Situation:** Enter a brief description of the background or context for the customer's dialogue (e.g., "The customer is calling about a delayed delivery and is very upset.").
* **Clear Chat:** Button to reset the conversation history.

### 2. Start Chatting (Main Panel)

1.  Enter your message in the text box at the bottom (this represents the **Customer**'s dialogue).
2.  Press **Enter** or click the send icon.
3.  The agent (MoodBud) processes the entire input prompt (`Emotion`, `Situation`, and `Customer` message) to generate an empathetic **Agent** response.

---

## ⚙️ Core Implementation Details

This section details the key technical components implemented in the PyTorch code.

### Data Preprocessing & Vocabulary

The `load_and_process_csv` function is responsible for preparing the raw text data:

* **Text Normalization:** Standardizes text by lowercasing, removing special characters, and cleaning dialogue tags.
* **Custom Tokenization:** Uses `re.findall` to preserve structural tokens (`"Emotion:"`, `"Situation:"`, etc.) alongside words and punctuation.
* **Vocabulary Creation:** Filters out low-frequency tokens (`min_freq=2`) to manage vocabulary size, then adds essential tokens:
    * Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
    * Emotion-specific tokens: e.g., `<emotion_anger>`

### Model Architecture

The chatbot utilizes a custom **Transformer Encoder-Decoder** model built with PyTorch, following the original 2017 architecture:

* **Embedding and Positional Encoding:** Converts token IDs into dense vectors and injects sequence order information via `PositionalEncoding`.
* **Encoder Layer:** Comprises **MultiHeadAttention** (Self-Attention) and a **PositionwiseFFN**, processing the input context.
* **Decoder Layer:** Includes three sub-layers: masked **Self-Attention** (for the target sequence), **Cross-Attention** (attending to the Encoder output), and a **PositionwiseFFN**.
* **`TransformerModel`:** Stacks the Encoder and Decoder layers and includes the final linear output layer to predict vocabulary logits.

### Generation Strategies

The application provides two methods for text generation at inference time:

1.  **`greedy_decode`**: A simple approach that selects the token with the highest probability at each step. It is fast but can get stuck in repetitive or sub-optimal sequences.

2.  **`beam_search`**: A more powerful heuristic search algorithm that keeps track of the $k$ most probable sequences (where $k$ is the `beam_size`) at each step. It further includes:
    * **No-Repeat N-gram Constraint:** Prevents the model from generating repetitive phrases (e.g., repeating the same 3-gram).
    * **Length Penalty:** A mechanism to favor slightly longer, more complete sequences over short ones, improving sentence flow and balance.