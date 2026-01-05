# Simple RNN for IMDB Sentiment Analysis

This project implements a sentiment analysis model for IMDB movie reviews using a Simple Recurrent Neural Network (RNN) built with TensorFlow/Keras. It includes training code, prediction examples, and an interactive web application for classifying movie reviews as positive or negative.

## Features

- **Model Training**: Jupyter notebook (`simplernn.ipynb`) for training a Simple RNN model on the IMDB dataset.
- **Prediction**: Example predictions in `prediction.ipynb` and a Streamlit web app in `main.py` for real-time sentiment analysis.
- **Pre-trained Model**: Saved model (`simple_rnn_imdb.keras`) ready for inference.
- **Interactive App**: User-friendly Streamlit interface for entering reviews and getting predictions.

## Installation

1. Clone or download this repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit App

To launch the interactive sentiment analysis web app:

```bash
streamlit run main.py
```

This will open a web browser where you can enter movie reviews and get instant sentiment predictions.

### Training the Model

Open `simplernn.ipynb` in Jupyter Notebook or VSCode to train the model from scratch. The notebook includes:

- Data loading and preprocessing
- Model architecture (Embedding + Simple RNN + Dense layers)
- Training with early stopping
- Model saving

### Making Predictions

Use `prediction.ipynb` for example predictions or integrate the prediction functions into your own code.

## Project Structure

- `main.py`: Streamlit web application for interactive sentiment analysis
- `simplernn.ipynb`: Jupyter notebook for training the Simple RNN model
- `prediction.ipynb`: Jupyter notebook with prediction examples
- `simple_rnn_imdb.keras`: Pre-trained model file
- `requirements.txt`: Python dependencies
- `.gitignore`: Git ignore file

## Requirements

- Python 3.7+
- TensorFlow
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Keras
- TensorBoard
- IPython Kernel

## Model Details

- **Dataset**: IMDB movie reviews (binary classification)
- **Architecture**: Embedding layer (10000 vocab, 128 dim) -> Simple RNN (128 units, ReLU) -> Dropout (0.5) -> Dense (1, sigmoid)
- **Max Sequence Length**: 300 (for training), 500 (for inference)
- **Vocabulary Size**: 10000 words

## License

This project is for educational purposes. Feel free to use and modify as needed.
