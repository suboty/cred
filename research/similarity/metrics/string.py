from typing import (
    Optional,
    Tuple,
    Callable,
    Union
)

from Levenshtein import (
    distance,
    ratio,
    hamming,
    jaro,
    jaro_winkler
)


class StringSimilarity:
    """
    Class for string similarity calculating.
    Build on python-Levenshtein library.
    See in https://pypi.org/project/python-Levenshtein/.
    """
    @staticmethod
    def get_distance(
            string1: str,
            string2: str,
            is_ratio: bool = False,
            processor: Optional[Callable] = None,
            score_cutoff: Optional[float] = None,
            weights: Optional[Tuple[int, int, int]] = None
    ) -> Union[int, float]:
        """
        Levenshtein distance
        :param string1: First string for comparison.
        :param string2: Second string for comparison.
        :param is_ratio: Is it ratio calculating?
        :param processor: Preprocessing function for strings.
        :param score_cutoff:
            For Distance:
                Maximum distance between string1 and string2, that is considered as a result.
                If the distance is bigger than score_cutoff, score_cutoff 1 is returned instead.
            For Ratio:
                Score threshold as a float between 0 and 1.0.
        :param weights: The weights for the three operations in the form (insertion, deletion, substitution).
            Default is (1, 1, 1)
        :return Levenshtein distance (int) or ratio (float).
        """
        kwargs = {}

        # optional params
        if score_cutoff:
            kwargs.setdefault('score_cutoff', score_cutoff)
        if weights and is_ratio:
            # There is no weights parameter in the ratio method
            kwargs.setdefault('weights', weights)
        if processor:
            kwargs.setdefault('processor', processor)

        if is_ratio:
            return ratio(
                string1,
                string2,
                **kwargs
            )

        return distance(
            string1,
            string2,
            **kwargs
        )

    @staticmethod
    def get_hamming_distance(
            string1: str,
            string2: str,
            pad: Optional[bool] = None,
            processor: Optional[Callable] = None,
            score_cutoff: Optional[float] = None,
    ) -> int:
        """
        Hamming distance
        :param string1: First string for comparison.
        :param string2: Second string for comparison.
        :param pad: Should strings be padded if there is a length difference.
        :param processor: Preprocessing function for strings.
        :param score_cutoff:
            Maximum distance between string1 and string2, that is considered as a result.
            If the distance is bigger than score_cutoff, score_cutoff 1 is returned instead
        :return Hamming distance (int).
        """
        kwargs = {}

        # optional params
        if score_cutoff:
            kwargs.setdefault('score_cutoff', score_cutoff)
        if pad:
            kwargs.setdefault('pad', pad)
        if processor:
            kwargs.setdefault('processor', processor)

        return hamming(
            string1,
            string2,
            **kwargs
        )

    @staticmethod
    def get_jaro_similarity(
            string1: str,
            string2: str,
            is_jaro_winkler: bool = False,
            prefix_weight: Optional[float] = None,
            processor: Optional[Callable] = None,
            score_cutoff: Optional[float] = None,
    ):
        """
        Jaro Similarity (or Jaro-Winkler)
        :param string1: First string for comparison.
        :param string2: Second string for comparison.
        :param is_jaro_winkler: Is it Jaro-Winkler similarity?
        :param processor: Preprocessing function for strings.
        :param score_cutoff:
            Optional argument for a score threshold as a float between 0 and 1.0.
            For ratio < score_cutoff 0 is returned instead.
        :param prefix_weight:
            Weight used for the common prefix of the two strings.
            Has to be between 0 and 0.25
        :return Jaro or Jaro-Winkler similarity (float).
        """
        kwargs = {}

        # optional params
        if score_cutoff:
            kwargs.setdefault('score_cutoff', score_cutoff)
        if prefix_weight and is_jaro_winkler:
            if prefix_weight > .25 or prefix_weight < 0:
                raise ValueError('Prefix Weight has to be between 0 and 0.25')
            kwargs.setdefault('prefix_weight', prefix_weight)
        if processor:
            kwargs.setdefault('processor', processor)

        if is_jaro_winkler:
            return jaro_winkler(
                string1,
                string2,
                **kwargs
            )

        return jaro(
            string1,
            string2,
            **kwargs
        )
