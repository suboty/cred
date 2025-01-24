import os
import warnings

from utils import *
from encoders.get_tf_idf_matrix import TfidfMatrix
from get_data import get_data_from_regex101
from utils import high_dimensional_visualization


if __name__ == '__main__':
    # disable warnings from scikit-learn and umap-learn
    warnings.filterwarnings("ignore")

    data, labels = get_data_from_regex101()

    # get data
    dataset = pd.DataFrame(data, columns=labels)
    dataset = dataset.loc[dataset['regex'] != '']

    list_of_regexes = dataset['regex'].tolist()
    dialects = dataset['dialect'].tolist()

    # random number for example printing
    random_n = random.randint(0, len(list_of_regexes))

    # prepare assets folder
    os.makedirs('assets', exist_ok=True)

    print(f'### Example of regexes:')
    [
        print(f"\t{i+1}) "+x)
        for i, x
        in enumerate(random.sample(list_of_regexes, 3))
    ]

    # get TF-IDF matrix (tokenize chars)
    print('### TF-IDF + Chars Tokenizing')
    chars_t, chars_m = TfidfMatrix.get_matrix_tokenize_by_chars(list_of_regexes)
    get_tf_idf_keywords(
        _tfidf_vectorizer=chars_t,
        _tfidf_matrix=chars_m,
        document_index=random_n,
        _list_of_regexes=list_of_regexes
    )
    high_dimensional_visualization(
        data=chars_m,
        name='tf_idf_chars',
        dialects=dialects,
    )

    # get TF-IDF matrix (tokenize non-terminals)
    print('### TF-IDF + Non-terminals Tokenizing')
    # select special characters, which used in regular expressions as a non-terminal symbols
    special_chars = []
    # from "space" to "slash"
    special_chars += get_all_unicode_letters('0020', '002F')
    # from "colon" to "at symbol"
    special_chars += get_all_unicode_letters('003A', '0040')
    # from "open square bracket" to "underscore"
    special_chars += get_all_unicode_letters('005B', '005F')
    # from "open curly bracket" to "tilda"
    special_chars += get_all_unicode_letters('007B', '007E')

    non_terminals_t, non_terminals_m = TfidfMatrix.get_matrix_tokenize_by_non_terminals(
        list_of_regexes=list_of_regexes,
        special_chars=special_chars
    )
    get_tf_idf_keywords(
        _tfidf_vectorizer=non_terminals_t,
        _tfidf_matrix=non_terminals_m,
        document_index=random_n,
        _list_of_regexes=list_of_regexes
    )
    high_dimensional_visualization(
        data=non_terminals_m,
        name='tf_idf_non_terminals',
        dialects=dialects,
    )

    # get TF-IDF matrix (tokenize with custom tokens)
    print('### TF-IDF + Custom Regex Tokenizing')
    tokens_t, tokens_m = TfidfMatrix.get_matrix_tokenize_by_regex_tokens(
        list_of_regexes
    )
    get_tf_idf_keywords(
        _tfidf_vectorizer=tokens_t,
        _tfidf_matrix=tokens_m,
        document_index=random_n,
        _list_of_regexes=list_of_regexes
    )
    high_dimensional_visualization(
        data=tokens_m,
        name='tf_idf_tokens',
        dialects=dialects,
    )
