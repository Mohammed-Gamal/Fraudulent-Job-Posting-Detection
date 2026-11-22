"""Stage 3b: a learned embedding layer for the text.

Why this exists
---------------
TF-IDF treats every term as an isolated dimension. `recruit` and `hiring` are
as unrelated as `recruit` and `aardvark`, and nothing is shared between them.
A learned embedding gives each vocabulary term a dense vector trained *for this
task*, so terms that play the same role in fraudulent adverts end up near each
other and the classifier can generalise across them.

Architecture
------------
    token ids (padded)                       (batch, seq_len)
      -> nn.Embedding(vocab, dim, padding_idx=0)   (batch, seq_len, dim)
      -> pooling over time, padding masked out     (batch, dim)
      -> concat structured features                (batch, dim + n_struct)
      -> Linear -> ReLU -> Dropout -> Linear       (batch, 1)
      -> BCEWithLogitsLoss(pos_weight=...)

Three details carry most of the weight here:

`padding_idx=0`
    Padding must not contribute a gradient. Without this the model learns an
    embedding for "nothing", and because short documents are mostly padding,
    that vector would dominate their representation.

`pos_weight`
    The imbalance treatment for a neural model. At a 4.84% positive rate an
    unweighted BCE loss is minimised by predicting "legitimate" everywhere.
    `pos_weight` scales the positive term of the loss by the inverse class
    ratio, which is the gradient-level equivalent of `class_weight="balanced"`
    used for the linear models -- so the comparison between them is fair.

Attention pooling
    Mean pooling dilutes a single damning sentence across 300 tokens.
    Additive attention learns which positions matter and, as a side effect,
    yields per-token weights that make the model's decision inspectable -- the
    embedding model's answer to the linear models' coefficients.

torch is an optional dependency. Importing this module without it succeeds;
only constructing the estimator fails, with an actionable message.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from . import config

logger = logging.getLogger(__name__)

# Reserved ids. 0 must be the pad so it can be handed to `padding_idx`.
PAD_ID = 0
UNK_ID = 1
RESERVED_TOKENS = ("<pad>", "<unk>")

POOLING_MODES: frozenset[str] = frozenset(
    {"mean", "max", "attention", "gated", "conv"}
)

# Kernel widths for convolutional pooling: unigram through 4-gram.
CONV_KERNELS: tuple[int, ...] = (1, 2, 3, 4)

TORCH_IMPORT_ERROR = (
    "PyTorch is required for the embedding model but is not installed.\n"
    "Install it with:  uv add torch\n"
    "Every other model in this project runs without it."
)

_TOKEN_SPLIT_RE = re.compile(r"\s+")


def _require_torch() -> Any:
    """Import torch on demand so the rest of the project never depends on it."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(TORCH_IMPORT_ERROR) from exc
    return torch


_TORCH_PRESENT: list[bool | None] = [None]


def torch_available() -> bool:
    """Whether the embedding model can be used in this environment.

    Checks that the package is importable without importing it. Importing
    torch takes about 2.5 seconds, and this is called while rendering the
    page -- just to decide whether to show one checkbox -- so the whole
    interface used to wait on it before painting anything. The real import
    happens when training starts, which is when that cost belongs.

    A package that is present but broken will still fail at import time, and
    `_require_torch` raises with an actionable message when it does.
    """
    if _TORCH_PRESENT[0] is None:
        from importlib.util import find_spec

        try:
            _TORCH_PRESENT[0] = find_spec("torch") is not None
        except (ImportError, ValueError):  # pragma: no cover - odd installs
            _TORCH_PRESENT[0] = False
    return bool(_TORCH_PRESENT[0])


# ---------------------------------------------------------------------------
# Text -> padded integer sequences
# ---------------------------------------------------------------------------
class SequenceEncoder(TransformerMixin, BaseEstimator):
    """Map cleaned text to fixed-length arrays of vocabulary ids.

    Fitted inside the cross-validation fold like every other transform, so the
    vocabulary is built from training documents only. A vocabulary built on the
    whole corpus would tell the model which terms exist in the test set -- the
    same leak as fitting TF-IDF before splitting, and just as easy to miss.

    Sequences are right-padded and truncated to `sequence_length`. Truncation
    keeps the head of the document: EMSCAD concatenates title, profile and
    description in that order, and the opening lines carry the pitch.
    """

    def __init__(
        self,
        max_vocabulary: int = 20_000,
        sequence_length: int = 300,
        min_document_frequency: int = 3,
    ) -> None:
        self.max_vocabulary = max_vocabulary
        self.sequence_length = sequence_length
        self.min_document_frequency = min_document_frequency

    # -- sklearn plumbing ---------------------------------------------------
    def _column(self, X: Any) -> list[str]:
        """Accept a DataFrame column, a 2-D single-column array, or a sequence."""
        if hasattr(X, "iloc"):  # Series or single-column DataFrame
            frame = X.iloc[:, 0] if getattr(X, "ndim", 1) == 2 else X
            return frame.fillna("").astype(str).tolist()
        array = np.asarray(X, dtype=object)
        if array.ndim == 2:
            if array.shape[1] != 1:
                raise ValueError(
                    f"SequenceEncoder expects one text column, got {array.shape[1]}"
                )
            array = array[:, 0]
        return ["" if value is None else str(value) for value in array]

    def fit(self, X: Any, y: Any = None) -> SequenceEncoder:
        documents = self._column(X)
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if self.max_vocabulary < len(RESERVED_TOKENS) + 1:
            raise ValueError(
                f"max_vocabulary must exceed {len(RESERVED_TOKENS)} reserved tokens"
            )

        # Document frequency, not term frequency: one posting repeating a word
        # forty times should not buy that word a place in the vocabulary.
        document_frequency: Counter[str] = Counter()
        for document in documents:
            tokens = _TOKEN_SPLIT_RE.split(document.strip())
            document_frequency.update({token for token in tokens if token})

        budget = self.max_vocabulary - len(RESERVED_TOKENS)
        frequent = [
            (token, count)
            for token, count in document_frequency.items()
            if count >= self.min_document_frequency
        ]
        # Sort by frequency, then alphabetically, so the vocabulary -- and
        # therefore every token id -- is reproducible across runs.
        frequent.sort(key=lambda item: (-item[1], item[0]))

        self.vocabulary_: dict[str, int] = {
            token: index for index, token in enumerate(RESERVED_TOKENS)
        }
        for token, _ in frequent[:budget]:
            self.vocabulary_[token] = len(self.vocabulary_)

        self.n_features_out_ = self.sequence_length
        self.document_frequency_ = document_frequency
        logger.info(
            "SequenceEncoder: vocabulary %s terms (%s above min_df=%s), seq_len %s",
            f"{len(self.vocabulary_):,}",
            f"{len(frequent):,}",
            self.min_document_frequency,
            self.sequence_length,
        )
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "vocabulary_")
        documents = self._column(X)
        length = self.sequence_length
        vocabulary = self.vocabulary_

        encoded = np.full((len(documents), length), PAD_ID, dtype=np.int64)
        for row, document in enumerate(documents):
            position = 0
            for token in _TOKEN_SPLIT_RE.split(document.strip()):
                if not token:
                    continue
                if position >= length:
                    break
                encoded[row, position] = vocabulary.get(token, UNK_ID)
                position += 1
        return encoded

    def inverse_tokens(self, ids: np.ndarray) -> list[str]:
        """Token strings for a single encoded row, padding dropped."""
        check_is_fitted(self, "vocabulary_")
        lookup = {index: token for token, index in self.vocabulary_.items()}
        return [lookup.get(int(i), "<unk>") for i in ids if int(i) != PAD_ID]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "vocabulary_")
        return np.asarray([f"token_{i}" for i in range(self.sequence_length)])

    # Output is a dense integer matrix; tell scikit-learn not to wrap it.
    def _more_tags(self) -> dict[str, Any]:  # pragma: no cover - sklearn<1.6
        return {"X_types": ["string"], "requires_y": False}

    def __sklearn_tags__(self):  # pragma: no cover - sklearn>=1.6
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.two_d_array = False
        return tags


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
def _build_module(
    torch: Any,
    vocabulary_size: int,
    embedding_dim: int,
    hidden_dim: int,
    n_structured: int,
    dropout: float,
    pooling: str,
    n_filters: int = 64,
) -> Any:
    """Construct the embedding network. Defined inside a function so that
    importing this module never touches torch."""
    nn = torch.nn

    class AttentionPool(nn.Module):
        """Additive attention over time, padding masked out.

        Returns the pooled vector and the attention weights, which are kept
        for the explanation panel in the interface.
        """

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.project = nn.Linear(dim, dim)
            self.score = nn.Linear(dim, 1, bias=False)

        def forward(self, hidden: Any, mask: Any) -> tuple[Any, Any]:
            scores = self.score(torch.tanh(self.project(hidden))).squeeze(-1)
            # Masked positions get a large negative score so softmax ignores
            # them. Using -inf produces NaN for an all-padding row.
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            # An all-padding row has uniform weights over nothing; zero it out
            # rather than returning the average of the padding embedding.
            weights = weights * mask.to(weights.dtype)
            pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
            return pooled, weights

    class GatedPool(nn.Module):
        """Weighted mean where each *vocabulary term* has one learned weight.

        Contextual attention has to learn a projection and a scoring vector
        from 690 positive examples, and on this dataset it overfits before it
        becomes useful. A single scalar per term is 20,000 parameters instead
        of 4,000-odd plus a matrix, cannot depend on position, and amounts to
        learning an importance weight per word -- a trained counterpart to the
        IDF weighting that makes TF-IDF work.

        It keeps the interpretability that mean pooling lacks: the weights are
        directly readable as "how much this model cares about this term".
        """

        def __init__(self, vocabulary: int) -> None:
            super().__init__()
            self.gate = nn.Embedding(vocabulary, 1, padding_idx=PAD_ID)
            # Start near zero so the first epoch is an unweighted mean and the
            # model departs from it only as evidence accumulates.
            nn.init.zeros_(self.gate.weight)

        def forward(self, hidden: Any, mask: Any, tokens: Any) -> tuple[Any, Any]:
            scores = self.gate(tokens).squeeze(-1)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=1) * mask.to(hidden.dtype)
            pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
            return pooled, weights

    class ConvPool(nn.Module):
        """Convolutional n-gram features, max-pooled over time.

        Every other pooling mode here is order-invariant: mean, max, attention
        and gated all produce the same vector for a document and for its
        shuffled tokens. That discards word order entirely, and with it the
        phrase signal -- "no experience", "use link", "earn weekly" -- that
        TF-IDF captures as bigrams and that turned out to be most of the gap
        between this model and the linear baseline.

        A convolution of width k slides a learned filter over k consecutive
        embeddings, so each filter detects an n-gram pattern wherever it
        occurs; max-over-time keeps the strongest activation for each filter.
        This is the standard convolutional sentence classifier (Kim, 2014) and
        it is the first architecture here that can represent a phrase.
        """

        def __init__(self, dim: int, n_filters: int, kernels: tuple[int, ...]) -> None:
            super().__init__()
            self.kernels = kernels
            self.convolutions = nn.ModuleList(
                [nn.Conv1d(dim, n_filters, kernel_size=k) for k in kernels]
            )
            self.output_dim = n_filters * len(kernels)

        def forward(self, hidden: Any, mask: Any) -> tuple[Any, Any]:
            # (batch, time, dim) -> (batch, dim, time) for Conv1d.
            signal = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).transpose(1, 2)
            pooled_parts, activation_maps = [], []
            for kernel, convolution in zip(self.kernels, self.convolutions, strict=True):
                if signal.shape[2] < kernel:
                    # Document shorter than the kernel: nothing to convolve.
                    pooled_parts.append(
                        torch.zeros(
                            signal.shape[0],
                            convolution.out_channels,
                            dtype=hidden.dtype,
                            device=hidden.device,
                        )
                    )
                    continue
                activated = torch.relu(convolution(signal))
                pooled_parts.append(activated.max(dim=2).values)
                activation_maps.append((kernel, activated))
            pooled = torch.cat(pooled_parts, dim=1)

            # Per-position importance for the explanation panel: how strongly
            # any filter fired at each position, spread back over the window
            # the filter covered.
            weights = torch.zeros(
                hidden.shape[0], hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            for kernel, activated in activation_maps:
                strength = activated.max(dim=1).values  # (batch, positions)
                for offset in range(kernel):
                    end = offset + strength.shape[1]
                    if end <= weights.shape[1]:
                        weights[:, offset:end] = torch.maximum(
                            weights[:, offset:end], strength
                        )
            weights = weights * mask.to(weights.dtype)
            total = weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            return pooled, weights / total

    class MaskedPool(nn.Module):
        """Mean or max pooling that ignores padding positions."""

        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

        def forward(self, hidden: Any, mask: Any) -> tuple[Any, Any]:
            expanded = mask.unsqueeze(-1).to(hidden.dtype)
            if self.mode == "mean":
                summed = (hidden * expanded).sum(dim=1)
                # clamp(min=1): an empty document would otherwise divide by 0.
                counts = expanded.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
            else:
                pooled = hidden.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
                pooled = torch.where(
                    mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled)
                )
            # Uniform weights over real tokens: no learned attention to report.
            weights = expanded.squeeze(-1) / expanded.sum(dim=1).clamp(min=1.0)
            return pooled, weights

    class EmbeddingNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                vocabulary_size, embedding_dim, padding_idx=PAD_ID
            )
            # Small init: large random embeddings saturate tanh in the
            # attention projection before any learning happens.
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
            with torch.no_grad():
                self.embedding.weight[PAD_ID].zero_()

            self.pooling_mode = pooling
            if pooling == "attention":
                self.pool = AttentionPool(embedding_dim)
                pooled_dim = embedding_dim
            elif pooling == "gated":
                self.pool = GatedPool(vocabulary_size)
                pooled_dim = embedding_dim
            elif pooling == "conv":
                self.pool = ConvPool(embedding_dim, n_filters, CONV_KERNELS)
                pooled_dim = self.pool.output_dim
            else:
                self.pool = MaskedPool(pooling)
                pooled_dim = embedding_dim

            # Dropout on the pooled text representation only. Applying it
            # after concatenation also zeroed the structured flags at random,
            # and `has_company_logo` is the single strongest feature in the
            # dataset -- dropping it 30% of the time threw away the signal the
            # linear models get intact, which is most of why the first version
            # of this model underperformed them.
            self.embedding_dropout = nn.Dropout(dropout)
            self.hidden = nn.Linear(pooled_dim + n_structured, hidden_dim)
            self.activation = nn.ReLU()
            self.head_dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_dim, 1)

        def forward(
            self, tokens: Any, structured: Any, return_attention: bool = False
        ) -> Any:
            mask = tokens != PAD_ID
            embedded = self.embedding(tokens)
            if self.pooling_mode == "gated":
                pooled, weights = self.pool(embedded, mask, tokens)
            else:
                pooled, weights = self.pool(embedded, mask)
            pooled = self.embedding_dropout(pooled)
            features = (
                torch.cat([pooled, structured], dim=1)
                if structured is not None and structured.shape[1] > 0
                else pooled
            )
            hidden = self.head_dropout(self.activation(self.hidden(features)))
            logits = self.output(hidden).squeeze(-1)
            if return_attention:
                return logits, weights
            return logits

    return EmbeddingNet()


# ---------------------------------------------------------------------------
# scikit-learn estimator
# ---------------------------------------------------------------------------
class EmbeddingTextClassifier(ClassifierMixin, BaseEstimator):
    """A neural classifier over a trainable embedding layer.

    Input layout
    ------------
    `fit`/`predict` take a dense 2-D array whose first `sequence_length`
    columns are token ids from `SequenceEncoder` and whose remaining columns
    are structured features. `build_embedding_pipeline` wires the two together
    so the split point cannot drift.

    This slightly awkward contract is what lets the model sit inside the same
    `ColumnTransformer` -> estimator pipeline as the linear baselines, and
    therefore be cross-validated by the same code with the same folds. A
    bespoke training loop outside the pipeline would not be comparable.
    """

    def __init__(
        self,
        sequence_length: int = 300,
        vocabulary_size: int = 20_000,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        pooling: str = "conv",
        n_filters: int = 64,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 128,
        max_epochs: int = 12,
        patience: int = 3,
        validation_fraction: float = 0.1,
        balanced: bool = True,
        tune_threshold: bool = True,
        random_state: int = config.RANDOM_STATE,
        n_threads: int | None = None,
        progress: Any = None,
        verbose: bool = False,
    ) -> None:
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pooling = pooling
        self.n_filters = n_filters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.balanced = balanced
        self.tune_threshold = tune_threshold
        self.random_state = random_state
        self.n_threads = n_threads
        self.progress = progress
        self.verbose = verbose

    # -- helpers ------------------------------------------------------------
    def _split_input(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {array.shape}")
        length = self.sequence_length
        if array.shape[1] < length:
            raise ValueError(
                f"input has {array.shape[1]} columns but sequence_length is "
                f"{length}; the encoder and the classifier disagree"
            )
        tokens = array[:, :length].astype(np.int64, copy=False)
        structured = array[:, length:].astype(np.float32, copy=False)
        return tokens, structured

    def _validate_parameters(self) -> None:
        if self.pooling not in POOLING_MODES:
            raise ValueError(
                f"pooling must be one of {sorted(POOLING_MODES)}, got {self.pooling!r}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def _report(self, fraction: float, label: str) -> None:
        if self.progress is not None:
            try:
                self.progress(fraction, label)
            except Exception:  # a broken reporter must not kill training
                logger.debug("progress callback failed", exc_info=True)

    # -- fit ----------------------------------------------------------------
    def fit(
        self, X: Any, y: Any, groups: Any = None
    ) -> EmbeddingTextClassifier:
        torch = _require_torch()
        self._validate_parameters()

        # Deterministic runs: the report quotes these numbers.
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if self.n_threads:
            torch.set_num_threads(int(self.n_threads))

        tokens, structured = self._split_input(X)
        labels = np.asarray(y).astype(np.float32).ravel()
        if tokens.shape[0] != labels.shape[0]:
            raise ValueError("X and y have different lengths")

        self.classes_ = np.unique(np.asarray(y))
        if self.classes_.size < 2:
            raise ValueError(
                "the embedding classifier needs both classes in the training data"
            )
        self.n_structured_ = int(structured.shape[1])

        # Vocabulary size must cover the largest id the encoder can emit.
        observed = int(tokens.max()) + 1 if tokens.size else len(RESERVED_TOKENS)
        vocabulary_size = max(int(self.vocabulary_size), observed)

        # Early-stopping holdout, carved out of the training data only. The
        # outer test split is never touched here.
        train_slice, valid_slice = self._holdout(labels, groups)

        device = torch.device("cpu")
        self.module_ = _build_module(
            torch,
            vocabulary_size=vocabulary_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_structured=self.n_structured_,
            dropout=self.dropout,
            pooling=self.pooling,
            n_filters=int(self.n_filters),
        ).to(device)

        token_tensor = torch.from_numpy(tokens)
        structured_tensor = torch.from_numpy(structured)
        label_tensor = torch.from_numpy(labels)

        # pos_weight = negatives / positives: the gradient-level counterpart of
        # class_weight="balanced" on the linear models.
        if self.balanced:
            positives = float(labels[train_slice].sum())
            negatives = float(len(train_slice) - positives)
            ratio = negatives / positives if positives > 0 else 1.0
        else:
            ratio = 1.0
        self.pos_weight_ = ratio
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(ratio, dtype=torch.float32)
        )
        optimiser = torch.optim.Adam(
            self.module_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        generator = np.random.default_rng(self.random_state)
        best_score, best_state, best_epoch = -np.inf, None, 0
        history: list[dict[str, float]] = []
        stalled = 0

        for epoch in range(1, self.max_epochs + 1):
            self.module_.train()
            order = generator.permutation(train_slice)
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                index = torch.from_numpy(np.ascontiguousarray(batch))
                optimiser.zero_grad(set_to_none=True)
                logits = self.module_(
                    token_tensor[index], structured_tensor[index]
                )
                loss = criterion(logits, label_tensor[index])
                loss.backward()
                # Attention plus a 1.3M-parameter embedding occasionally
                # produces a large gradient on a rare token; clipping keeps
                # one batch from wrecking the run.
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), 5.0)
                optimiser.step()
                total_loss += float(loss.detach())
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)
            score = self._validation_score(
                torch, token_tensor, structured_tensor, labels, valid_slice
            )
            history.append(
                {"epoch": epoch, "train_loss": round(train_loss, 5),
                 "val_avg_precision": round(score, 5)}
            )
            if self.verbose:
                logger.info(
                    "epoch %2d/%d  loss=%.4f  val_AP=%.4f",
                    epoch, self.max_epochs, train_loss, score,
                )
            self._report(
                epoch / self.max_epochs,
                f"epoch {epoch}/{self.max_epochs} (loss {train_loss:.3f})",
            )

            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.module_.state_dict().items()
                }
                stalled = 0
            else:
                stalled += 1
                if self.patience and stalled >= self.patience:
                    logger.info(
                        "early stop at epoch %d (best epoch %d, val AP %.4f)",
                        epoch, best_epoch, best_score,
                    )
                    break

        if best_state is not None:
            self.module_.load_state_dict(best_state)
        self.module_.eval()

        self.history_ = history
        self.best_epoch_ = best_epoch
        self.best_validation_score_ = float(best_score)
        self.vocabulary_size_ = vocabulary_size

        # Where to cut the logits. `pos_weight` deliberately inflates the loss
        # on the minority class, which shifts the logit scale: zero is no
        # longer the F1-optimal boundary, and scoring the model there
        # understates it. The threshold is chosen on the early-stopping
        # holdout -- data this model never trained on -- so it costs nothing
        # extra and the test split is still untouched.
        self.threshold_ = 0.0
        if self.tune_threshold and len(valid_slice) > 0:
            self.threshold_ = self._tuned_threshold(
                torch, token_tensor, structured_tensor, labels, valid_slice
            )

        self._report(1.0, "embedding model trained")
        return self

    def _tuned_threshold(
        self,
        torch: Any,
        token_tensor: Any,
        structured_tensor: Any,
        labels: np.ndarray,
        valid_slice: np.ndarray,
    ) -> float:
        """F1-optimal logit threshold on the held-out validation slice."""
        from sklearn.metrics import precision_recall_curve

        held_out = labels[valid_slice]
        if np.unique(held_out).size < 2:
            return 0.0
        scores = self._raw_scores(torch, token_tensor, structured_tensor, valid_slice)
        precision, recall, thresholds = precision_recall_curve(held_out, scores)
        precision, recall = precision[:-1], recall[:-1]
        if thresholds.size == 0:
            return 0.0
        total = precision + recall
        f1 = np.divide(
            2 * precision * recall, total, out=np.zeros_like(total), where=total > 0
        )
        chosen = float(thresholds[int(np.argmax(f1))])
        logger.info(
            "tuned logit threshold %.3f (holdout F1=%.4f)", chosen, float(f1.max())
        )
        return chosen

    def _holdout(
        self, labels: np.ndarray, groups: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Indices for training and for the early-stopping validation slice.

        When a group array is supplied the holdout is group-disjoint, so a
        duplicated description cannot sit on both sides and flatter the
        validation score that decides when to stop.
        """
        n = len(labels)
        positions = np.arange(n)
        if self.validation_fraction <= 0:
            return positions, positions

        if groups is not None:
            from sklearn.model_selection import StratifiedGroupKFold

            n_splits = max(2, int(round(1 / self.validation_fraction)))
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            try:
                train_slice, valid_slice = next(
                    splitter.split(positions.reshape(-1, 1), labels, np.asarray(groups))
                )
                return train_slice, valid_slice
            except ValueError:
                logger.debug("grouped holdout failed; falling back to stratified")

        stratify = labels if np.unique(labels).size > 1 else None
        try:
            train_slice, valid_slice = train_test_split(
                positions,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            # Too few positives to stratify: fall back to a plain split.
            train_slice, valid_slice = train_test_split(
                positions,
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
        return train_slice, valid_slice

    def _validation_score(
        self,
        torch: Any,
        token_tensor: Any,
        structured_tensor: Any,
        labels: np.ndarray,
        valid_slice: np.ndarray,
    ) -> float:
        """Average precision on the holdout; ranking quality, not accuracy.

        Average precision rather than loss or accuracy because it is what
        model selection uses everywhere else in this project, and because
        accuracy on a 4.84% positive rate is uninformative.
        """
        from sklearn.metrics import average_precision_score

        if len(valid_slice) == 0:
            return 0.0
        held_out = labels[valid_slice]
        scores = self._raw_scores(torch, token_tensor, structured_tensor, valid_slice)
        if np.unique(held_out).size < 2:
            return 0.0
        return float(average_precision_score(held_out, scores))

    def _raw_scores(
        self,
        torch: Any,
        token_tensor: Any,
        structured_tensor: Any,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        self.module_.eval()
        index = (
            torch.from_numpy(np.ascontiguousarray(rows))
            if rows is not None
            else None
        )
        tokens = token_tensor if index is None else token_tensor[index]
        structured = (
            structured_tensor if index is None else structured_tensor[index]
        )
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, tokens.shape[0], max(self.batch_size, 1)):
                stop = start + max(self.batch_size, 1)
                logits = self.module_(tokens[start:stop], structured[start:stop])
                outputs.append(logits.detach().numpy().ravel())
        if not outputs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outputs)

    # -- predict ------------------------------------------------------------
    def decision_function(self, X: Any) -> np.ndarray:
        """Logits. Zero is the decision boundary, matching LinearSVC's margin."""
        torch = _require_torch()
        self._check_fitted()
        tokens, structured = self._split_input(X)
        return self._raw_scores(
            torch, torch.from_numpy(tokens), torch.from_numpy(structured)
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        logits = self.decision_function(X)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> np.ndarray:
        """Hard labels at the fitted threshold.

        `decision_function` still returns raw logits, so ranking metrics
        (average precision, ROC AUC) are unaffected by the threshold and
        remain directly comparable with the other models.
        """
        self._check_fitted()
        threshold = float(getattr(self, "threshold_", 0.0))
        return (self.decision_function(X) >= threshold).astype(int)

    def _check_fitted(self) -> None:
        if not hasattr(self, "module_"):
            raise NotFittedError(
                "this EmbeddingTextClassifier is not fitted yet; call fit first"
            )

    # -- interpretability ---------------------------------------------------
    def attention_weights(self, X: Any) -> np.ndarray:
        """Per-position pooling weights for each row of `X`.

        With attention pooling these are learned and informative. With mean or
        max pooling they are uniform over the real tokens, which the caller
        should treat as "no attribution available" rather than as evidence.
        """
        torch = _require_torch()
        self._check_fitted()
        tokens, structured = self._split_input(X)
        self.module_.eval()
        with torch.no_grad():
            _, weights = self.module_(
                torch.from_numpy(tokens),
                torch.from_numpy(structured),
                return_attention=True,
            )
        return weights.detach().numpy()

    def nearest_terms(
        self,
        encoder: SequenceEncoder,
        term: str,
        top_n: int = 8,
        min_document_frequency: int = 25,
    ) -> list[dict[str, Any]]:
        """Terms whose learned embeddings are closest to `term` by cosine.

        Restricted to terms appearing in at least `min_document_frequency`
        documents. A term seen twice in 14,000 training documents receives a
        gradient twice, so its vector is still essentially its random
        initialisation -- and random vectors in 64 dimensions are all roughly
        equidistant, which produced neighbour lists full of one-off tokens
        that looked like findings and were noise.

        Read the result with care even after filtering. These embeddings are
        trained to make the classifier's job easy, not to model similarity, so
        neighbours reflect "used the same way in fraudulent adverts" rather
        than "means the same thing". At this corpus size the geometry is often
        not interpretable at all, which is itself worth reporting.
        """
        self._check_fitted()
        vocabulary = getattr(encoder, "vocabulary_", None)
        if not vocabulary or term not in vocabulary:
            return []

        frequency = getattr(encoder, "document_frequency_", {})
        weights = self.module_.embedding.weight.detach().numpy()
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        normalised = weights / np.clip(norms, 1e-9, None)
        target = normalised[vocabulary[term]]
        similarity = normalised @ target

        inverse = {index: token for token, index in vocabulary.items()}
        order = np.argsort(-similarity)
        results: list[dict[str, Any]] = []
        for index in order:
            position = int(index)
            if position in (PAD_ID, UNK_ID) or position == vocabulary[term]:
                continue
            token = inverse.get(position)
            if token is None:
                continue
            if frequency.get(token, 0) < min_document_frequency:
                continue
            results.append(
                {
                    "term": token,
                    "similarity": round(float(similarity[position]), 4),
                    "documents": int(frequency.get(token, 0)),
                }
            )
            if len(results) >= top_n:
                break
        return results

    # -- persistence --------------------------------------------------------
    def _module_config(self) -> dict[str, Any]:
        """Arguments needed to rebuild this estimator's network from scratch."""
        return {
            "vocabulary_size": int(getattr(self, "vocabulary_size_", self.vocabulary_size)),
            "embedding_dim": int(self.embedding_dim),
            "hidden_dim": int(self.hidden_dim),
            "n_structured": int(getattr(self, "n_structured_", 0)),
            "dropout": float(self.dropout),
            "pooling": str(self.pooling),
            "n_filters": int(self.n_filters),
        }

    def __getstate__(self) -> dict[str, Any]:
        """Persist weights, not the network object.

        Two things here cannot be pickled:

        `progress` is supplied by the web interface and closes over the job
        registry, which holds a `threading.Lock`.

        The network classes are defined inside `_build_module` so that
        importing this file never touches torch. That makes them local
        classes, which pickle cannot find by reference.

        Together these meant the winning model was lost at the very last step
        of training, surfacing later as an EOFError on load -- nowhere near
        the cause. Saving the `state_dict` plus the arguments needed to
        rebuild the network is also the way PyTorch itself recommends
        persisting a model, and it survives changes to the class internals.
        """
        state = dict(self.__dict__)
        state["progress"] = None
        module = state.pop("module_", None)
        if module is not None:
            state["_module_state"] = {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            }
            state["_module_config"] = self._module_config()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        weights = state.pop("_module_state", None)
        config_ = state.pop("_module_config", None)
        self.__dict__.update(state)
        self.progress = None
        if weights is None or config_ is None:
            return
        torch = _require_torch()
        self.module_ = _build_module(torch, **config_)
        self.module_.load_state_dict(weights)
        self.module_.eval()

    def __sklearn_tags__(self):  # pragma: no cover - sklearn>=1.6
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        return tags
