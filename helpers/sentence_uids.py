import pandas as pd
import string
import re

class SentenceMapper:
    def __init__(self, df_mapping):
        """
        Initialize with mapping DataFrame.
        :param df_mapping: DataFrame with 'expression' and 'uid' columns.
        """
        self.df_mapping = df_mapping
        self.unmapped_words = set()

    def process_sentence(self, sentence):
        """Lowercase, remove possessive 's, remove punctuation, and tokenize the sentence."""
        # Lowercase and remove possessive 's using regex
        processed_sentence = re.sub(r"'s\b", "", sentence.lower())
        # Remove punctuation
        processed_sentence = processed_sentence.translate(str.maketrans('', '', string.punctuation))
        return self._tokenize_sentence(processed_sentence)

    def _tokenize_sentence(self, sentence):
        """Custom tokenization to handle phrases like 'climb up' and 'climb down'."""
        # Define the phrases that should be treated as single words
        phrases = ["climb up", "climb down", "ice cream"]

        # Replace the phrases with a placeholder to avoid splitting them
        for phrase in phrases:
            sentence = sentence.replace(phrase, phrase.replace(' ', '_'))  # Replace with underscores

        # Tokenize the sentence
        return sentence.split()

    def map_words_to_uids(self, sentence):
        """Map each word in the sentence to its UID(s) and return the result as a list of UIDs."""
        tokenized_sentence = self.process_sentence(sentence)
        uids = []

        for word in tokenized_sentence:
            # Check if the word contains underscores (for phrases like 'climb_up')
            if '_' in word:
                # Replace underscores with spaces to match the original phrase
                original_word = word.replace('_', ' ')
            else:
                original_word = word

            # Find the UID using the original word (with spaces)
            matched_uids = self.df_mapping[self.df_mapping['expression'] == original_word]['uid'].tolist()
            if matched_uids:
                uids.append(matched_uids[0])  # Use the first UID found
            else:
                self.unmapped_words.add(original_word)  # Track unmapped words if needed

        return uids


def get_uid(sentence):
    # Example mapping DataFrame
    df_mapping = pd.read_csv('/DATA7/vaibhav/tokenization/CISLR/helpers/filtered_vocab_Blimp_CISLR.csv')

    # Initialize the mapper
    mapper = SentenceMapper(df_mapping)

    # Map words to UIDs for the sentence
    result = mapper.map_words_to_uids(sentence)
    return result

