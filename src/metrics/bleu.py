from collections.abc import Sequence
from typing import Any, Optional, Union

from torch import Tensor
from torchmetrics.functional.text.bleu import _bleu_score_update
from torchmetrics.functional.text.sacre_bleu import (
    _SacreBLEUTokenizer,
    _TokenizersLiteral,
)
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SacreBLEUScore.plot"]


class SacreBLEUScore(BLEUScore):
    """Calculate BLEU score of machine translated text with one or more references.

    This implementation follows the behaviour of SacreBLEU. The SacreBLEU implementation differs from the NLTK BLEU
    implementation in tokenization techniques.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (Sequence): An iterable of machine translated corpus
    - ``target`` (Sequence): An iterable of iterables of reference corpus

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``sacre_bleu`` (torch.Tensor): A tensor with the SacreBLEU Score

    Args:
        n_gram: Gram value ranged from 1 to 4
        smooth: Whether to apply smoothing, see SacreBLEU
        tokenize: Tokenization technique to be used. Choose between ``'none'``, ``'13a'``, ``'zh'``, ``'intl'``,
            ``'char'``, ``'ja-mecab'``, ``'ko-mecab'``, ``'flores101'`` and ``'flores200'``.
        lowercase:  If ``True``, BLEU score over lowercased text is calculated.
        kwargs: Additional keyword arguments, see 'Metric kwargs' for more info.
        weights:
            Weights used for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used.

    Raises:

        ValueError:
            If ``tokenize`` not one of 'none', '13a', 'zh', 'intl' or 'char'
        ValueError:
            If ``tokenize`` is set to 'intl' and `regex` is not installed
        ValueError:
            If a length of a list of weights is not ``None`` and not equal to ``n_gram``.

    Example:
        >>> from torchmetrics.text import SacreBLEUScore
        >>> preds = ['the cat is on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> sacre_bleu = SacreBLEUScore()
        >>> sacre_bleu(preds, target)
        tensor(0.7598)

    References:
        - Automatic Evaluation of Machine Translation Quality Using Longest Common Subsequence
          and Skip-Bigram Statistics by Chin-Yew Lin and Franz Josef Och
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(
        self,
        n_gram: int = 1,
        smooth: bool = False,
        tokenize: _TokenizersLiteral = "13a",
        lowercase: bool = True,
        weights: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_gram=n_gram, smooth=smooth, weights=weights, **kwargs)
        self.tokenizer = _SacreBLEUTokenizer(tokenize, lowercase)

    def update(self, preds: Sequence[str], target: Sequence[Sequence[str]]) -> None:
        """Update state with predictions and targets."""
        self.preds_len, self.target_len = _bleu_score_update(
            preds,
            target,
            self.numerator,
            self.denominator,
            self.preds_len,
            self.target_len,
            self.n_gram,
            self.tokenizer,
        )

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Generate visualization plot.

        Creates a visualization of the BLEU score trends.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            matplotlib.figure.Figure and matplotlib.axes.Axes: The generated plot

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        Example:
            >>> # Example plotting a single value
            >>> from torchmetrics.text import SacreBLEUScore
            >>> metric = SacreBLEUScore()
            >>> preds = ['the cat is on the mat']
            >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
            >>> metric.update(preds, target)
            >>> figure, axes = metric.plot()

            >>> # Example plotting multiple values
            >>> from torchmetrics.text import SacreBLEUScore
            >>> metric = SacreBLEUScore()
            >>> preds = ['the cat is on the mat']
            >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> figure, axes = metric.plot(values)
        """
        return self._plot(val, ax)
