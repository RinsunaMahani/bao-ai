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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectorizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True,
                        help="Required: the classifier is trained on encoded "
                             "labels, so this is what maps them back to names.")
    parser.add_argument("--out", type=Path,
                        default=Path("models/language_detector_pan_african.joblib"))
    args = parser.parse_args()

    import joblib
    from sklearn.pipeline import Pipeline

    vectorizer = pickle.loads(args.vectorizer.read_bytes())
    classifier = pickle.loads(args.model.read_bytes())
    encoder = pickle.loads(args.encoder.read_bytes())

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
