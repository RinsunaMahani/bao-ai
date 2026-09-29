"""Combine the three exported .pkl files into one loadable bundle.

    python scripts/build_pan_african_bundle.py \
        --vectorizer tfidf_vectorizer.pkl \
        --model african_lang_model.pkl \
        --encoder label_encoder.pkl

The Colab notebook exports the vectoriser, the classifier and the label
encoder separately. The app loads a single joblib file containing a fitted
Pipeline and its labels, so that three files cannot drift apart or arrive
half-copied.

The label mapping needs care. The classifier was trained on ENCODED labels,
so `classifier.classes_` is [0, 1, 2, ...], not language names — reading
names off it directly yields integers. But `predict_proba` columns still
follow `classes_`, so the correct name for column i is
`encoder.classes_[classifier.classes_[i]]`, composing the two rather than
trusting either alone. Getting this wrong mislabels every prediction
without raising anything, which is the same class of bug that hit the
language detector and the sign classifier.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def reexport(path: Path) -> int:
    """Re-saves the bundle under the scikit-learn that is installed now.

    A joblib bundle records the version that pickled it, and loading it
    under a different one emits InconsistentVersionWarning on every
    estimator — four per load here, which drowned the test output at 280
    warnings and would hide a real one. The warning is not merely noise:
    unpickling an estimator across versions is not guaranteed to be
    faithful, and when it is not, it fails silently.

    This is a round trip of the already-fitted estimators, not a retrain,
    so it needs none of the Colab exports — which is the point, because
    those are not in this repository. It is only safe because the caller
    VERIFIES predictions are unchanged afterwards; see
    tests/test_pan_african.py. Re-saving a bundle that was already being
    mis-unpickled would bake the damage in, so the check is the feature,
    not the re-save.

    Run this after upgrading scikit-learn, then run the tests.
    """
    import joblib
    import sklearn

    if not path.exists():
        print(f"  ! no bundle at {path}", file=sys.stderr)
        return 1

    bundle = joblib.load(path)
    joblib.dump({"pipeline": bundle["pipeline"], "labels": bundle["labels"]}, path)
    print(f"  re-saved {path} under scikit-learn {sklearn.__version__}")
    print("  now run: python -m pytest tests/test_pan_african.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectorizer", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--encoder", type=Path,
                        help="Required when building: the classifier is trained "
                             "on encoded labels, so this maps them back to names.")
    parser.add_argument("--out", type=Path,
                        default=Path("models/language_detector_pan_african.joblib"))
    parser.add_argument("--reexport", action="store_true",
                        help="Re-save the EXISTING bundle under the installed "
                             "scikit-learn, instead of building one from the "
                             "three Colab exports.")
    args = parser.parse_args()

    import joblib
    from sklearn.pipeline import Pipeline

    if args.reexport:
        return reexport(args.out)

    missing = [n for n in ("vectorizer", "model", "encoder") if getattr(args, n) is None]
    if missing:
        print(f"  ! --{' --'.join(missing)} required when building "
              f"(or pass --reexport)", file=sys.stderr)
        return 1

    # The project's own training outputs, named on the command line by the
    # person building the bundle - never a file from a user or a download.
    vectorizer = pickle.loads(args.vectorizer.read_bytes())  # noqa: S301
    classifier = pickle.loads(args.model.read_bytes())  # noqa: S301
    encoder = pickle.loads(args.encoder.read_bytes())  # noqa: S301

    # Column i of predict_proba corresponds to classifier.classes_[i], which
    # is an ENCODED label; the encoder turns that back into a name.
    labels = [str(encoder.classes_[int(code)]) for code in classifier.classes_]

    pipeline = Pipeline([("tfidf", vectorizer), ("clf", classifier)])

    # Prove it works before writing anything.
    probabilities = pipeline.predict_proba(["test"])[0]
    if len(probabilities) != len(labels):
        print("  ! probability width does not match label count — refusing to write.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "labels": labels}, args.out)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"  wrote {args.out} ({size_mb:.1f} MB, {len(labels)} classes)")
    print(f"  labels: {', '.join(labels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
