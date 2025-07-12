# Data Challenge - Political Speech Analysis

This project analyzes German political speeches using natural language processing techniques including text summarization and sentiment analysis.

## Overview

The notebook processes a dataset of German political speeches (`speakers_and_parties_with_metadata.csv`) and performs the following analyses:
- Text summarization using German-specific models
- Sentiment analysis of political speeches
- Visualization of sentiment trends across speeches

## Features

### 1. Data Processing
- Loads political speech data with speaker, party, and session information
- Restructures data to group speeches by party and parliamentary session
- Creates summaries per party and session combination

### 2. Text Summarization
- Uses `deutsche-telekom/mt5-small-sum-de-mit-v1` model for German text summarization
- Processes speech content to generate concise summaries
- Handles long German political texts with proper truncation

### 3. Sentiment Analysis
- Employs `oliverguhr/german-sentiment-bert` for German sentiment classification
- Analyzes sentiment of summarized speeches
- Provides both sentiment labels (positive/negative/neutral) and confidence scores

### 4. Data Visualization
- Creates line plots showing sentiment scores over time
- Displays sentiment trends for specific parties and sessions
- Includes summary statistics (e.g., negative sentiment count)

## Requirements

```bash
pip install pandas
pip install tqdm
pip install transformers
pip install torch
pip install matplotlib
pip install numpy
pip install ipdb
```

## Data Structure

The input CSV file should contain the following columns:
- `Speaker`: Name of the person giving the speech
- `Party`: Political party affiliation
- `Sitzung`: Parliamentary session number
- `Speech content`: Full text of the speech

## Usage

1. **Setup and Data Loading**
   ```python
   df = pd.read_csv("speakers_and_parties_with_metadata.csv")
   ```

2. **Data Preprocessing**
   - Extract unique parties and sessions
   - Create structured dataframe with grouped speeches

3. **Text Summarization**
   ```python
   # Example: Summarize AfD speeches from session 81
   afd_session_81 = summarized_content_per_party_and_session(new_df, 'AfD', 81)
   ```

4. **Sentiment Analysis**
   ```python
   # Apply sentiment analysis to summaries
   afd_session_81['sentiment_label'] = afd_session_81['summary'].progress_apply(label_per_sentence)
   afd_session_81['sentiment_score'] = afd_session_81['summary'].progress_apply(score_per_sentence)
   ```

5. **Visualization**
   ```python
   # Plot sentiment trends
   plt.plot(y_pos, afd_session_81.sentiment_score.iloc[0])
   ```

## Key Functions

### Data Processing Functions
- `get_parties(df)`: Extract unique political parties from dataset
- `get_sessions(df)`: Extract unique parliamentary sessions
- `get_summary_per_sessions(party, session, df)`: Group speeches by party and session
- `create_new_dataframe(parties, sessions)`: Create structured dataframe

### NLP Functions
- `sumarize_german_text(text, max_input_length=None)`: Summarize German text using mT5 model
- `label_per_sentence(x)`: Extract sentiment labels for text list
- `score_per_sentence(x)`: Extract sentiment confidence scores for text list

## Models Used

1. **Summarization Model**: `deutsche-telekom/mt5-small-sum-de-mit-v1`
   - Multilingual T5 model fine-tuned for German summarization
   - Optimized for German political and formal texts

2. **Sentiment Analysis Model**: `oliverguhr/german-sentiment-bert`
   - BERT model specifically trained on German text
   - Classifies text as positive, negative, or neutral

## Output

The analysis provides:
- Summarized versions of political speeches
- Sentiment classifications and confidence scores
- Visual representations of sentiment trends
- Statistical summaries (e.g., count of negative sentiments)

## Example Analysis

The notebook includes a complete analysis of AfD (Alternative for Germany) speeches from parliamentary session 81, demonstrating:
- Text summarization of political content
- Sentiment trend visualization
- Statistical summary of sentiment distribution

## GPU Requirements

The models are configured to run on CUDA-enabled GPUs for faster processing. Ensure you have:
- CUDA-compatible GPU
- PyTorch with CUDA support installed

## Notes

- The notebook includes debugging tools (`ipdb`) for development
- Progress bars (`tqdm`) provide feedback during long-running operations
- CSS styling is included for better output formatting in Jupyter notebooks
- All text processing is optimized for German language content

## License

This project is designed for academic and research purposes in political speech analysis.
