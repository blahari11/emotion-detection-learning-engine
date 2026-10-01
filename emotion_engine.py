import re
import math

EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

KEYWORDS = {
    "Bored": {
        "boring": 2.5, "bored": 3.0, "dull": 2.5, "not interested": 3.0,
        "repetitive": 2.0, "tired of": 2.2, "sleepy": 1.5
    },
    "Confident": {
        "understand": 1.7, "clear now": 3.0, "i can solve": 3.0, "easy": 2.0,
        "confident": 3.0, "got it": 3.0, "i know": 2.0, "ready": 1.5
    },
    "Confused": {
        "confused": 3.0, "don't understand": 3.0, "do not understand": 3.0,
        "lost": 2.7, "unclear": 2.5, "not clear": 2.5, "what does": 1.5,
        "can't understand": 3.0, "cannot understand": 3.0, "stuck": 1.8
    },
    "Curious": {
        "why": 1.8, "how": 1.5, "what if": 2.5, "curious": 3.0,
        "want to know": 2.7, "wonder": 2.5, "interested": 1.7, "explore": 1.8
    },
    "Frustrated": {
        "frustrated": 3.0, "irritated": 3.0, "angry": 2.7, "hate": 2.5,
        "giving up": 3.0, "give up": 3.0, "many times": 1.8, "still can't": 3.0,
        "still cannot": 3.0, "again and again": 2.5, "annoying": 2.7
    },
}

class EmotionEngine:
    """
    Hybrid inference engine.

    The BiLSTM side is represented by bidirectional sequence/context scoring so the
    repository runs immediately without a trained checkpoint. train_bilstm.py
    contains a real PyTorch BiLSTM architecture for dataset-based training.

    The BERT side uses a Hugging Face text-classification pipeline when available.
    If the model cannot be downloaded, a contextual fallback keeps the demo usable.
    """
    def __init__(self):
        self.bert_pipe = None
        try:
            from transformers import pipeline
            self.bert_pipe = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
        except Exception:
            self.bert_pipe = None

    def _keyword_scores(self, text):
        t = text.lower()
        scores = {e: 0.15 for e in EMOTIONS}
        for emotion, phrases in KEYWORDS.items():
            for phrase, weight in phrases.items():
                if phrase in t:
                    scores[emotion] += weight
        if "?" in text:
            scores["Curious"] += 0.7
        if re.search(r"\b(still|again|tried)\b", t):
            scores["Frustrated"] += 0.6
        if re.search(r"\b(don't|cannot|can't|unable)\b", t):
            scores["Confused"] += 0.8
        return scores

    def _normalize(self, scores):
        exps = {k: math.exp(min(v, 8)) for k, v in scores.items()}
        total = sum(exps.values())
        return {k: v / total for k, v in exps.items()}

    def _bilstm_style_predict(self, text):
        # Bidirectional context enhancement: forward and reversed token windows.
        base = self._keyword_scores(text)
        tokens = re.findall(r"[a-z']+", text.lower())
        forward = " ".join(tokens)
        backward = " ".join(reversed(tokens))
        if "tried" in forward and ("understand" in forward or "solve" in forward):
            base["Frustrated"] += 1.0
        if "understand" in backward and ("not" in backward or "don't" in forward):
            base["Confused"] += 0.8
        probs = self._normalize(base)
        emotion = max(probs, key=probs.get)
        return {"emotion": emotion, "confidence": probs[emotion], "scores": probs}

    def _bert_predict(self, text):
        scores = {e: 0.02 for e in EMOTIONS}
        if self.bert_pipe:
            try:
                raw = self.bert_pipe(text)[0]
                mapping = {
                    "joy": "Confident",
                    "surprise": "Curious",
                    "sadness": "Confused",
                    "fear": "Confused",
                    "anger": "Frustrated",
                    "disgust": "Bored",
                    "neutral": "Bored",
                }
                for item in raw:
                    mapped = mapping.get(item["label"].lower())
                    if mapped:
                        scores[mapped] += float(item["score"])
            except Exception:
                scores = self._keyword_scores(text)
        else:
            scores = self._keyword_scores(text)

        # Rule enhancement aligns generic emotion labels with learning context.
        keyword = self._keyword_scores(text)
        for e in EMOTIONS:
            scores[e] += keyword[e] * 0.45
        probs = self._normalize(scores)
        emotion = max(probs, key=probs.get)
        return {"emotion": emotion, "confidence": probs[emotion], "scores": probs}

    def predict(self, text):
        bilstm = self._bilstm_style_predict(text)
        bert = self._bert_predict(text)
        keyword = self._normalize(self._keyword_scores(text))

        combined = {}
        for emotion in EMOTIONS:
            combined[emotion] = (
                bilstm["scores"][emotion] * 0.35 +
                bert["scores"][emotion] * 0.45 +
                keyword[emotion] * 0.20
            )

        final_emotion = max(combined, key=combined.get)
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        mixed = [{"emotion": e, "score": s} for e, s in ranked[:2] if s >= 0.12]
        all_scores = [{"emotion": e, "score": combined[e]} for e in EMOTIONS]

        return {
            "bilstm": bilstm,
            "bert": bert,
            "final_emotion": final_emotion,
            "mixed_emotions": mixed,
            "all_scores": all_scores,
        }
