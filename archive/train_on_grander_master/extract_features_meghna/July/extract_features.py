import numpy as np
import os
import json
import nltk
import string
import textstat
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, sent_tokenize
from textblob import TextBlob
import spacy
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nrclex import NRCLex
from collections import Counter
import numpy as np
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import spacy
import textstat
from textblob import TextBlob
from simpletransformers.ner import NERModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import pipeline
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Load MPQA Lexicon with Correct Parsing
mpqa_lexicon = {
    "certainty_adverbs": set(),
    "certainty_verbs": set(),
    "certainty_adjectives": set(),
    "doubt_adverbs": set(),
    "doubt_verbs": set(),
    "doubt_adjectives": set(),
    "hedges": set(),
    "emphatics": set(),
    "possibility_modals": set(),
    "necessity_modals": set(),
    "predictive_modals": set(),
}

# Comprehensive Hedge List (if MPQA is missing them)
manual_hedge_list = {"seems", "appears", "possibly", "probably", "apparently", 
                     "somewhat", "kind of", "sort of", "allegedly", "presumably", 
                     "maybe", "perhaps", "suggests", "could be"}

possibility_modals = {"might", "could", "may", "can", "perhaps", "possibly", "potentially"}
necessity_modals = {"must", "should", "ought", "need", "required", "necessary", "essential"}
predictive_modals = {"will", "shall", "going", "bound", "likely", "certain"}

# Open MPQA File
with open("subjclueslen1-HLTEMNLP05.tff", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        
        # Initialize default values
        word, pos_tag, strength, polarity = None, None, None, None  

        # Extract attributes from each line
        for part in parts:
            if part.startswith("word1="):
                word = part.split("=")[1]
            elif part.startswith("pos1="):
                pos_tag = part.split("=")[1]
            elif part.startswith("type="):
                strength = part.split("=")[1]
            elif part.startswith("priorpolarity="):
                polarity = part.split("=")[1]

        # Ensure all necessary fields exist
        if not word or not pos_tag or not strength or not polarity:
            continue

        # Certainty-related words (strongly subjective & positive/neutral polarity)
        if strength == "strongsubj" and polarity in ["positive", "neutral"]:
            if pos_tag in ["adv", "anypos"]:
                mpqa_lexicon["certainty_adverbs"].add(word)
            elif pos_tag == "verb":
                mpqa_lexicon["certainty_verbs"].add(word)
            elif pos_tag == "adj":
                mpqa_lexicon["certainty_adjectives"].add(word)

        # Doubt-related words (weakly subjective & neutral polarity)
        elif strength == "weaksubj" and polarity == "neutral":
            if pos_tag in ["adv", "anypos"]:
                mpqa_lexicon["doubt_adverbs"].add(word)
            elif pos_tag == "verb":
                mpqa_lexicon["doubt_verbs"].add(word)
            elif pos_tag == "adj":
                mpqa_lexicon["doubt_adjectives"].add(word)

        # Expanded Hedge Detection
        if strength == "weaksubj" and pos_tag in ["adv", "anypos"]:
            mpqa_lexicon["hedges"].add(word)

        elif word in manual_hedge_list:  # Manually add common hedging words
            mpqa_lexicon["hedges"].add(word)

        # Emphatics (Strongly subjective & negative polarity words)
        elif strength == "strongsubj" and polarity == "negative":
            mpqa_lexicon["emphatics"].add(word)

        # Assign modal verbs to correct categories
        if word in possibility_modals:
            mpqa_lexicon["possibility_modals"].add(word)
        elif word in necessity_modals:
            mpqa_lexicon["necessity_modals"].add(word)
        elif word in predictive_modals:
            mpqa_lexicon["predictive_modals"].add(word)

# Print the number of words loaded for each category
print("✅ Loaded MPQA Lexicon Word Counts:", {k: len(v) for k, v in mpqa_lexicon.items()})


emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)
hedge_classifier = pipeline(
    "text-classification",
    model="ChrisLiewJY/BERTweet-Hedge",
    return_all_scores=True,
    # These extra args will be passed to the tokenizer
    truncation=True,
    max_length=128
)

hedge_model = NERModel(
    'bert',
    'jeniakim/hedgehog',
    use_cuda=True,
    labels=["C", "D", "E", "I", "N"]
)

def get_hedge_features(text):
    predictions, _ = hedge_model.predict([text])
    categories = ["C", "D", "E", "I", "N"]

    counts = {cat: 0 for cat in categories}
    total_tokens = 0

    # Process predictions
    for sentence in predictions:
        for token_dict in sentence:
            for _, label in token_dict.items():
                counts[label] += 1
                total_tokens += 1

    # Calculate ratios
    ratios = {f"{cat}_ratio": counts[cat] / total_tokens if total_tokens else 0 for cat in categories}

    # Merge counts and ratios
    features = {f"hedge_{cat}_count": counts[cat] for cat in categories}
    features.update(ratios)

    return features

# Assume mpqa_lexicon is defined globally with the required keys
# e.g., mpqa_lexicon = {"certainty_adverbs": set([...]), "certainty_verbs": set([...]), ... }

def extract_all_features(text):
    from nltk import pos_tag
    """ Extracts lexical, syntactic, readability, sentiment, emotion, and MPQA-based features from text. """
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    doc = nlp(text)
    
    # Lexical Features
    word_count = len(words)
    sentence_count = max(len(sentences), 1)  # Avoid division by zero
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    std_word_length = np.std([len(word) for word in words]) if words else 0
    type_token_ratio = len(set(words)) / word_count if word_count > 0 else 0
    hapax_legomena = sum(1 for word, count in Counter(words).items() if count == 1) / word_count if word_count > 0 else 0
    stopword_ratio = sum(1 for word in words if word.lower() in stop_words) / word_count if word_count > 0 else 0
    punctuation_density = sum(1 for char in text if char in string.punctuation) / word_count if word_count > 0 else 0

    # POS Tagging
    pos_tags = [tag for _, tag in pos_tag(words)]
    pos_counts = Counter(pos_tags)
    noun_ratio = sum(pos_counts.get(tag, 0) for tag in ['NN', 'NNS', 'NNP']) / word_count if word_count > 0 else 0
    verb_ratio = sum(pos_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']) / word_count if word_count > 0 else 0
    adj_ratio = sum(pos_counts.get(tag, 0) for tag in ['JJ', 'JJR', 'JJS']) / word_count if word_count > 0 else 0
    adv_ratio = sum(pos_counts.get(tag, 0) for tag in ['RB', 'RBR', 'RBS']) / word_count if word_count > 0 else 0
    pronoun_ratio = sum(pos_counts.get(tag, 0) for tag in ['PRP', 'PRP$']) / word_count if word_count > 0 else 0

    # Syntactic Features
    dependency_depths = [token.dep_ for token in doc if token.dep_]
    avg_dependency_depth = len(dependency_depths) / len(doc) if len(doc) > 0 else 0
    function_word_count = sum(1 for token in doc if token.is_stop)
    function_word_ratio = function_word_count / len(doc) if len(doc) > 0 else 0
    punctuation_usage = sum(1 for char in text if char in "!?.") / word_count if word_count > 0 else 0

    # Readability & Complexity
    readability_score = textstat.flesch_kincaid_grade(text) if text else 0
    sentence_length_variation = np.std([len(sent.split()) for sent in sentences]) if sentences else 0
    subordinate_clause_ratio = sum(1 for token in doc if token.dep_ in ["mark", "advcl", "ccomp", "xcomp"]) / word_count if word_count > 0 else 0

    # Sentiment Analysis using TextBlob
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity

    # Emotion Analysis using Hugging Face pipeline
    results = emotion_classifier(text)
    if results and isinstance(results[0], list):
        results = results[0]
    # Convert the list of dicts to a dictionary mapping each label to its score
    emotion_dict = {result['label'].lower(): result['score'] for result in results}
    anger = emotion_dict.get("anger", 0)
    joy = emotion_dict.get("joy", 0)
    fear = emotion_dict.get("fear", 0)
    sadness = emotion_dict.get("sadness", 0)
    disgust = emotion_dict.get("disgust", 0)
    surprise = emotion_dict.get("surprise", 0)

    # MPQA-Based Features
    text_tokens = set(word.lower() for word in words)  # Convert text to lowercase for matching
    certainty_adverbs_count = len(text_tokens.intersection(mpqa_lexicon["certainty_adverbs"]))
    certainty_verbs_count = len(text_tokens.intersection(mpqa_lexicon["certainty_verbs"]))
    certainty_adjectives_count = len(text_tokens.intersection(mpqa_lexicon["certainty_adjectives"]))
    doubt_adverbs_count = len(text_tokens.intersection(mpqa_lexicon["doubt_adverbs"]))
    doubt_verbs_count = len(text_tokens.intersection(mpqa_lexicon["doubt_verbs"]))
    doubt_adjectives_count = len(text_tokens.intersection(mpqa_lexicon["doubt_adjectives"]))
    #hedges_count = len(text_tokens.intersection(mpqa_lexicon["hedges"]))
    hedge_results = hedge_classifier(text, truncation=True, max_length=128)
    if hedge_results and isinstance(hedge_results[0], list):
        hedge_results = hedge_results[0]
    # Extract the score for the hedge label. (Adjust label name as needed—usually it should be something like "Hedge")
    hedges_score = next((item["score"] if item["score"] > 0.5 else 0 for item in hedge_results if item["label"] == "LABEL_1"), 0)
    # possibility_modals_count = len(text_tokens.intersection(mpqa_lexicon["possibility_modals"]))
    # necessity_modals_count = len(text_tokens.intersection(mpqa_lexicon["necessity_modals"]))
    # predictive_modals_count = len(text_tokens.intersection(mpqa_lexicon["predictive_modals"]))
    hedge_features = get_hedge_features(text)
    
    final_features = [
        word_count, sentence_count, avg_word_length, std_word_length, type_token_ratio, hapax_legomena,
        stopword_ratio, punctuation_density, noun_ratio, verb_ratio, adj_ratio, adv_ratio, pronoun_ratio,
        avg_dependency_depth, function_word_ratio, punctuation_usage, readability_score,
        sentence_length_variation, subordinate_clause_ratio, sentiment_polarity, sentiment_subjectivity,
        anger, joy, fear, sadness, disgust, surprise,
        certainty_adverbs_count, certainty_verbs_count, certainty_adjectives_count,
        doubt_adverbs_count, doubt_verbs_count, doubt_adjectives_count,
        hedges_score,
        hedge_features["hedge_C_count"], hedge_features["hedge_D_count"], hedge_features["hedge_E_count"],
        hedge_features["hedge_I_count"], hedge_features["hedge_N_count"],
        hedge_features["C_ratio"], hedge_features["D_ratio"], hedge_features["E_ratio"],
        hedge_features["I_ratio"], hedge_features["N_ratio"]
    ]

    return pd.Series(final_features)


def process_and_save_features(file):
    """ 
    Loads one dataset, extracts all features (including emotion & MPQA‐based features), 
    and saves them for visualization.
    """
    from nltk import pos_tag, word_tokenize, sent_tokenize

    # Load CSV file
    df = pd.read_csv(file)
    #df = df.head(10).copy()

    # Ensure necessary columns exist
    if 'sentence' not in df.columns or 'Status' not in df.columns:
        raise ValueError("CSV file must contain 'sentence' and 'Status' columns.")

    # Copy the DataFrame and extract features
    df_features = df.copy()
    df_features[feature_columns] = df['sentence'].apply(extract_all_features)

    # Construct an output file name (e.g., "mydata.csv" → "mydata_processed.csv")
    output_file = file.replace(".csv", "_processed.csv")
    df_features.to_csv(output_file, index=False)

    print("Feature extraction including MPQA markers and emotions completed and saved to", output_file)
    
    
    
def visualize_and_save_selected_features_single(df, label, selected_features, save_path="plots/feature_comparison_single.png"):
    """ 
    Generates a figure with subplots for each selected feature in a single dataset,
    ensuring the same scale for comparison.
    
    Parameters:
      - df: DataFrame, dataset with extracted features.
      - label: str, label for the dataset.
      - selected_features: list of feature names to visualize.
      - save_path: str, path to save the final figure.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    num_features = len(selected_features)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(10, 4 * num_features))
    
    # In case there's only one feature, make axes iterable
    if num_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(selected_features):
        # Skip if the feature is missing or all zeros/NaNs
        if feature not in df.columns:
            print(f"⚠️ Warning: Feature '{feature}' not found in the dataset. Skipping...")
            continue
        if df[feature].isnull().all():
            print(f"⚠️ Warning: Feature '{feature}' contains only NaN values. Skipping...")
            continue
        if df[feature].sum() == 0:
            print(f"⚠️ Warning: Feature '{feature}' contains only zeros. Skipping...")
            continue

        # Determine binning strategy
        min_val = df[feature].min()
        max_val = df[feature].max()

        if feature in ["certainty_adverbs_count", "certainty_verbs_count", "certainty_adjectives_count",
                       "doubt_adverbs_count", "doubt_verbs_count", "doubt_adjectives_count",
                       "hedges_score",
                       "possibility_modals_count", "necessity_modals_count", "predictive_modals_count"]:
            bins = range(int(min_val), int(max_val) + 2)
        else:
            bins = np.linspace(min_val, max_val, 30)
        
        # Create the histogram for this feature
        sns.histplot(df[feature], bins=bins, kde=True, ax=axes[i])
        axes[i].set_title(f'{feature} Distribution ({label})')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlim(min_val, max_val)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")
    plt.show()
    
def process_file_for_visualization(file, feature_type="all"):
    """ 
    Loads a single file, selects features based on feature_type, and visualizes them.
    """
    import pandas as pd

    # Load CSV file
    df = pd.read_csv(file)

    # Ensure necessary columns exist
    if 'text' not in df.columns or 'stance' not in df.columns:
        raise ValueError("CSV file must contain 'text' and 'stance' columns.")

    # Define the feature categories
    feature_categories = {
    "lexical": [
        "word_count", "sentence_count", "avg_word_length", "type_token_ratio",
        "hapax_legomena", "stopword_ratio", "punctuation_density",
        "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pronoun_ratio"
    ],
    "syntactic": [
        "avg_dependency_depth", "function_word_ratio", "punctuation_usage"
    ],
    "readability": [
        "readability_score", "sentence_length_variation", "subordinate_clause_ratio"
    ],
    "sentiment": [
        "sentiment_polarity", "sentiment_subjectivity"
    ],
    "emotion": [
        "anger", "joy", "fear", "sadness", "disgust", "surprise"
    ],
    "mpqa": [
        "certainty_adverbs_count", "certainty_verbs_count", "certainty_adjectives_count",
        "doubt_adverbs_count", "doubt_verbs_count", "doubt_adjectives_count"
    ],
    "hedging": ["hedges_score",
        "hedge_C_count", "hedge_D_count", "hedge_E_count",
        "hedge_I_count", "hedge_N_count", 
        "C_ratio", "D_ratio", "E_ratio", "I_ratio", "N_ratio"
    ],
    "all": [
        "word_count", "sentence_count", "avg_word_length", "type_token_ratio",
        "hapax_legomena", "stopword_ratio", "punctuation_density", "noun_ratio", "verb_ratio",
        "adj_ratio", "adv_ratio", "pronoun_ratio", "avg_dependency_depth", "function_word_ratio",
        "punctuation_usage", "readability_score", "sentence_length_variation",
        "subordinate_clause_ratio", "sentiment_polarity", "sentiment_subjectivity",
        "anger", "joy", "fear", "sadness", "disgust", "surprise",
        "certainty_adverbs_count", "certainty_verbs_count", "certainty_adjectives_count",
        "doubt_adverbs_count", "doubt_verbs_count", "doubt_adjectives_count",
        "hedges_score", 
        "hedge_C_count", "hedge_D_count", "hedge_E_count",
        "hedge_I_count", "hedge_N_count", 
        "C_ratio", "D_ratio", "E_ratio", "I_ratio", "N_ratio"
    ]
}


    if feature_type not in feature_categories:
        raise ValueError("Invalid feature_type. Choose from 'lexical', 'syntactic', 'readability', 'sentiment', 'emotion', 'mpqa', or 'all'.")
    selected_features = feature_categories[feature_type]

    # Call your visualization function for a single file.
    visualize_and_save_selected_features_single(df, file, selected_features)

feature_columns = [
    "word_count", "sentence_count", "avg_word_length", "std_word_length", "type_token_ratio", "hapax_legomena",
    "stopword_ratio", "punctuation_density", "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pronoun_ratio",
    "avg_dependency_depth", "function_word_ratio", "punctuation_usage", "readability_score",
    "sentence_length_variation", "subordinate_clause_ratio", "sentiment_polarity", "sentiment_subjectivity",
    "anger", "joy", "fear", "sadness", "disgust", "surprise",
    "certainty_adverbs_count", "certainty_verbs_count", "certainty_adjectives_count",
    "doubt_adverbs_count", "doubt_verbs_count", "doubt_adjectives_count",
    "hedges_score", "hedge_C_count", "hedge_D_count", "hedge_E_count",
    "hedge_I_count", "hedge_N_count",
    "C_ratio", "D_ratio", "E_ratio", "I_ratio", "N_ratio"
]

import os

def process_all_csv_files(directory):
    skipped_files = []

    for root, dirs, files in os.walk(directory):
        if 'overall' in root.split(os.sep):
            continue
        for file in files:
            if file.lower().endswith('.csv') and file.lower() != 'f1_results.csv' and 'examples' not in file:
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                if not df.empty:
                    print(f"Processing {file_path}")
                    process_and_save_features(file_path)
                else:
                    print(f"Skipping empty file: {file_path}")
                    skipped_files.append(file_path)

    if skipped_files:
        with open("skipped_files_pstance.txt", "w") as f:
            for skipped_file in skipped_files:
                f.write(f"{skipped_file}\n")

# Example usage
if __name__ == "__main__":
    #target_directory = "/home/p/parush/style_markers/classifications/covid"  # Replace with your directory path
    #process_all_csv_files(target_directory)
    #process_and_save_features('final_correct_examples_berts.csv')
    process_and_save_features('final_misclassified_examples_berts.csv')