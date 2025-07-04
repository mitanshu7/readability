# 📚 Text Readability Control System

An intelligent text generation and classification system that maintains readability levels while incorporating specific vocabulary constraints. The system uses machine learning models to classify text readability and employs Large Language Models (LLMs) to generate text that matches specific readability criteria.

## 🚀 Features

- **Readability Classification**: Automatically classify text into Elementary, Intermediate, or Advanced reading levels
- **Controlled Text Generation**: Generate text that maintains specific readability levels while incorporating required vocabulary
- **Multi-Modal ML Pipeline**: Combines text embeddings and traditional readability metrics for enhanced classification
- **Interactive Web Interface**: User-friendly Gradio interface for exploring generated text samples
- **Flexible LLM Support**: Works with both local and API-based language models
- **Validation Framework**: Automatic validation of generated text against readability and vocabulary requirements

## 🏗️ Architecture

The system consists of several key components:

### Core Components
- **`main.py`**: Main text generation pipeline with recursive validation
- **`generate.py`**: LLM inference utilities (local and API-based)
- **`validate.py`**: Text validation using trained ML models
- **`app.py`**: Gradio web interface for interactive demonstrations

### Data Processing Pipeline
- **`utils/dataset/`**: Dataset processing utilities for the OneStopEnglish corpus
- **`utils/model/`**: Machine learning model training scripts

### Models
- **SVM Classifiers**: Various Support Vector Machine models for readability classification
- **Embedding Models**: Sentence transformer models for text representation
- **Combined Models**: Hybrid approaches using both embeddings and readability metrics

## 📊 Dataset

The system is built on the [OneStopEnglish](https://github.com/nishkalavallabhi/OneStopEnglishCorpus) dataset, which provides:
- **Three Reading Levels**: Elementary, Intermediate, and Advanced
- **Parallel Texts**: Same content adapted for different reading levels
- **Rich Annotations**: Readability scores and linguistic features

## 🛠️ Installation

### Prerequisites
- Python 3.12 or higher
- GPU support (optional, for local LLM inference)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd readability

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and model preferences
```

### Environment Configuration
Create a `.env` file with the following variables:
```bash
# LLM Configuration
LOCAL=true/false              # Use local model or API
LLM_MODEL=model_name         # Model name (e.g., "meta-llama/Llama-3.2-3B-Instruct")
API_URL=your_api_url         # API endpoint (if using API)
API_KEY=your_api_key         # API key (if using API)
```

## 🚀 Usage

### Web Interface
Launch the interactive Gradio interface:
```bash
python app.py
```
Access the interface at `http://localhost:7895`

### Text Generation Pipeline
Generate text with controlled readability:
```bash
python main.py
```

### Model Training
Train new classification models:
```bash
# Generate embeddings
python utils/dataset/OneStopEnglish_embed.py

# Calculate readability scores
python utils/dataset/OneStopEnglish_readability.py

# Train SVM classifiers
python utils/model/OneStopEnglish_embed_readability_SVM.py
```

## 🎯 Key Features

### 1. Readability Classification
The system uses multiple approaches for readability classification:
- **Text Embeddings**: Using sentence transformers for semantic understanding
- **Readability Metrics**: Traditional metrics (Flesch-Kincaid, SMOG, etc.)
- **Hybrid Models**: Combining embeddings with readability scores

### 2. Controlled Text Generation
- **Readability Preservation**: Maintains the original text's reading level
- **Vocabulary Constraints**: Ensures inclusion of specified "trip words"
- **Iterative Refinement**: Recursive validation and regeneration until criteria are met

### 3. Trip Words System
"Trip words" are challenging vocabulary items selected based on:
- **Reading Level**: Different word length thresholds for each level
- **Complexity**: Longer words for higher reading levels
- **Context Integration**: Natural incorporation into generated text

### 4. Validation Framework
Comprehensive validation ensures generated text meets requirements:
- **Readability Validation**: Uses trained ML models to verify reading level
- **Vocabulary Validation**: Confirms inclusion of all required trip words
- **Feedback System**: Provides specific feedback for regeneration

## 📁 Project Structure

```
readability/
├── main.py                 # Main generation pipeline
├── generate.py             # LLM inference utilities
├── validate.py             # Text validation framework
├── app.py                  # Gradio web interface
├── pyproject.toml          # Project configuration
├── datasets/
│   └── OneStopEnglish/     # Processed datasets
├── models/
│   └── OneStopEnglish/     # Trained ML models
└── utils/
    ├── dataset/            # Data processing scripts
    └── model/              # Model training scripts
```

## 🔧 Technical Details

### Machine Learning Models
- **SVM Variants**: Linear, RBF, Polynomial, and Sigmoid kernels
- **Feature Engineering**: Multiple scaling and dimensionality reduction techniques
- **Hybrid Approaches**: Combining embeddings with traditional readability metrics

### LLM Integration
- **Local Models**: Support for Hugging Face transformers
- **API Models**: OpenAI-compatible API integration
- **Flexible Configuration**: Easy switching between local and API-based models

### Validation Metrics
- **Flesch Reading Ease**
- **Flesch-Kincaid Grade Level**
- **SMOG Index**
- **Automated Readability Index**
- **Coleman-Liau Index**
- **Dale-Chall Readability Score**
- **Gunning Fog Index**

## 🎨 Web Interface

The Gradio interface provides:
- **Random Sample Generation**: Explore generated text examples
- **Original vs Generated**: Side-by-side comparison
- **Readability Information**: Display of reading level and trip words
- **Model Information**: Current LLM configuration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.



## 🔗 References

- [OneStopEnglish Corpus](https://github.com/nishkalavallabhi/OneStopEnglishCorpus)
- [Sentence Transformers](https://www.sbert.net/)
- [Gradio Documentation](https://gradio.app/)

---

**Note**: For UMAP processing, use Python 3.9 with the command:
```bash
uvx --with umap-learn,pandas,pyarrow --python 3.9 python
```
This avoids compatibility issues with newer Python versions.
