import os
import argparse
import warnings

from utils import *
from encoders.get_tf_idf_matrix import TfidfMatrix
from encoders.get_bert_embeddings import BertEmbeddings
from get_data import get_data_from_regex101
from algorithms.kmeans import KMeansAlgorithm
from utils import (
    high_dimensional_visualization, 
    get_experiment_name, 
    make_clustering_report
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cred-clustering')

    # is need verbose print
    parser.add_argument(
        '-v', '--verbose',
        action='store_true'
    )
    # is need update of visualization
    parser.add_argument(
        '-u', '--update',
        action='store_true'
    )
    # encoder
    parser.add_argument('--algname', type=str, default='bert')
    # filter word for getting data
    parser.add_argument('--filter', type=str, default=None)

    args = parser.parse_args()

    km = KMeansAlgorithm()

    # disable warnings from scikit-learn and umap-learn
    warnings.filterwarnings("ignore")

    data, labels = get_data_from_regex101(args.filter)

    print(f'### Work with {len(data)} samples')

    # get data
    dataset = pd.DataFrame(data, columns=labels)
    dataset = dataset.loc[dataset['regex'] != '']

    list_of_regexes = dataset['regex'].tolist()
    dialects = dataset['dialect'].tolist()

    # random number for example printing
    random_n = random.randint(0, len(list_of_regexes))

    if args.verbose:
        print(f'### Example of regexes:')
        [
            print(f"\t{i+1}) "+x)
            for i, x
            in enumerate(random.sample(list_of_regexes, 3))
        ]

    os.makedirs('clustering_reports', exist_ok=True)

    # prepare assets folder
    if 'tf_idf' in args.algname:
        # get TF-IDF matrix (tokenize chars)
        print('### TF-IDF + Chars Tokenizing')
        exp_name = get_experiment_name(
            alg_name='tf_idf_chars',
            filter_word=args.filter
        )
        savepath = Path('assets', exp_name)
        os.makedirs(savepath, exist_ok=True)

        chars_t, chars_m = TfidfMatrix.get_matrix_tokenize_by_chars(list_of_regexes)
        if args.verbose:
            get_tf_idf_keywords(
                _tfidf_vectorizer=chars_t,
                _tfidf_matrix=chars_m,
                document_index=random_n,
                _list_of_regexes=list_of_regexes
            )
        pca, umap = high_dimensional_visualization(
            data=chars_m,
            name='tf_idf_chars',
            dialects=dialects,
            n_neighbors=50,
            umap_min_dist=0.25,
            savepath=savepath,
        )

        km(
            data=chars_m,
            pipeline_name='tf_idf_chars',
            verbose=False,
            savepath=savepath,
            data_2d=umap
        )

        make_clustering_report(
            experiment_name=exp_name,
            encoder='tf_idf_chars',
            clustering='kmeans++',
            img_savepath=Path('..', savepath),
            savepath='clustering_reports',
            filter_word=args.filter
        )

        # get TF-IDF matrix (tokenize non-terminals)
        print('### TF-IDF + Non-terminals Tokenizing')
        exp_name = get_experiment_name(
            alg_name='tf_idf_non_terminals',
            filter_word=args.filter
        )
        savepath = Path('assets', exp_name)
        os.makedirs(savepath, exist_ok=True)

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
        if args.verbose:
            get_tf_idf_keywords(
                _tfidf_vectorizer=non_terminals_t,
                _tfidf_matrix=non_terminals_m,
                document_index=random_n,
                _list_of_regexes=list_of_regexes
            )
        pca, umap = high_dimensional_visualization(
            data=non_terminals_m,
            name='tf_idf_non_terminals',
            dialects=dialects,
            n_neighbors=50,
            umap_min_dist=0.25,
            savepath=savepath,
        )

        km(
            data=non_terminals_m,
            pipeline_name='tf_idf_non_terminals',
            verbose=False,
            savepath=savepath,
            data_2d=umap
        )

        make_clustering_report(
            experiment_name=exp_name,
            encoder='tf_idf_non_terminals',
            clustering='kmeans++',
            img_savepath=Path('..', savepath),
            savepath='clustering_reports',
            filter_word=args.filter
        )

        # get TF-IDF matrix (tokenize with custom tokens)
        print('### TF-IDF + Custom Regex Tokenizing')
        exp_name = get_experiment_name(
            alg_name='tf_idf_tokens',
            filter_word=args.filter
        )
        savepath = Path('assets', exp_name)
        os.makedirs(savepath, exist_ok=True)

        tokens_t, tokens_m = TfidfMatrix.get_matrix_tokenize_by_regex_tokens(
            list_of_regexes
        )
        if args.verbose:
            get_tf_idf_keywords(
                _tfidf_vectorizer=tokens_t,
                _tfidf_matrix=tokens_m,
                document_index=random_n,
                _list_of_regexes=list_of_regexes
            )
        pca, umap = high_dimensional_visualization(
            data=tokens_m,
            name='tf_idf_tokens',
            dialects=dialects,
            n_neighbors=50,
            umap_min_dist=0.25,
            savepath=savepath,
        )

        km(
            data=tokens_m,
            pipeline_name='tf_idf_tokens',
            verbose=False,
            savepath=savepath,
            data_2d=umap
        )

        make_clustering_report(
            experiment_name=exp_name,
            encoder='tf_idf_tokens',
            clustering='kmeans++',
            img_savepath=Path('..', savepath),
            savepath='clustering_reports',
            filter_word=args.filter
        )

    elif 'bert' in args.algname:
        # create BERT inference
        be = BertEmbeddings()

        exp_name = get_experiment_name(
            alg_name=be.__repr__(),
            filter_word=args.filter
        )
        savepath = Path('assets', exp_name)
        os.makedirs(savepath, exist_ok=True)

        # get BERT embeddings (bert_base_uncased)
        print(f'### BERT embeddings ({be.name})')
        embeddings = be.get_bert_regex(list_of_regexes)
        
        pca, umap = high_dimensional_visualization(
            data=embeddings,
            name=be.name,
            dialects=dialects,
            n_neighbors=50,
            umap_min_dist=0.25,
            savepath=savepath,
        )

        km(
            data=embeddings,
            pipeline_name=be.name,
            verbose=False,
            savepath=savepath,
            data_2d=umap
        )

        make_clustering_report(
            experiment_name=exp_name,
            encoder=be.__repr__(),
            clustering='kmeans++',
            img_savepath=Path('..', savepath),
            savepath='clustering_reports',
            filter_word=args.filter
        )
