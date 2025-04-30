import argparse

from get_data import get_data_from_database


queries = {
    'all': 0,
    'similar': 1,
    'same_construction': 2,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cred-clustering')

    # regex source
    parser.add_argument('--regexSource', type=str, default='regex101')

    # regex group (query name)
    parser.add_argument('--regexGroup', type=str, default='all')

    # regex construction for same_construction grouping
    parser.add_argument('--regexConstruction', type=str, default=None)

    # init objects
    args = parser.parse_args()

    if args.regexConstruction:
        data = get_data_from_database(
            database=args.regexSource,
            query_index=queries.get(args.regexGroup),
            kwargs_construction=args.regexConstruction
        )
    else:
        data = get_data_from_database(
            database=args.regexSource,
            query_index=queries.get(args.regexGroup),
        )

    print(data)
