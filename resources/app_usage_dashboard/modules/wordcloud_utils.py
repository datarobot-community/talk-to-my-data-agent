import traceback

import streamlit as st
from streamlit.logger import get_logger
from wordcloud import WordCloud

# For Japanese tokenization, use janome
try:
    from janome.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False

# For TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = get_logger("streamlit")


def generate_user_wordcloud(text_data, font_path, current_language, _):
    if not text_data:
        st.warning(_("warnings.no_data_for_wordcloud"))
        return
    processed_text = text_data
    if current_language == "ja":
        if JANOME_AVAILABLE:
            try:
                words = [token.surface for token in tokenizer.tokenize(text_data)]
                processed_text = " ".join(words)
            except Exception as e:
                st.error(_("errors.janome_tokenize_error", error=str(e)))
                return
        else:
            st.error(_("errors.janome_init_error"))
            return
    # Count word frequencies
    from collections import Counter

    word_freq = Counter(processed_text.split())
    try:
        wordcloud = WordCloud(
            font_path=font_path, width=800, height=400, background_color="white"
        ).generate_from_frequencies(word_freq)
        st.image(wordcloud.to_array())
    except RuntimeError as e:
        st.error(_("errors.wordcloud_font_error", font_path=font_path, e=str(e)))
    except Exception as e:
        st.error(_("errors.wordcloud_unexpected_error", e=str(e)))


def generate_error_wordcloud(text_list, font_path, current_language, _):
    if not text_list or not any(text_list):
        st.warning(_("warnings.no_data_for_wordcloud"))
        return
    if current_language == "ja":
        if JANOME_AVAILABLE:
            try:
                # Tokenize each error message
                text_list = [
                    " ".join([token.surface for token in tokenizer.tokenize(t)])
                    for t in text_list
                ]
            except Exception as e:
                st.error(_("errors.janome_tokenize_error", error=str(e)))
                logger.error(_("errors.janome_tokenize_error", error=str(e)))
                logger.error(traceback.format_exc())
                return
        else:
            st.error(_("errors.janome_init_error"))
            return
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is required for TF-IDF word cloud.")
        return
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
        tfidf_matrix = vectorizer.fit_transform(text_list)
        scores = tfidf_matrix.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()
        word_scores = dict(zip(words, scores))
        wordcloud = WordCloud(
            font_path=font_path, width=800, height=400, background_color="white"
        ).generate_from_frequencies(word_scores)
        st.image(wordcloud.to_array())
    except RuntimeError as e:
        st.error(_("errors.wordcloud_font_error", font_path=font_path, e=str(e)))
    except Exception as e:
        st.error(_("errors.wordcloud_unexpected_error", e=str(e)))
