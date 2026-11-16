
import re


class ContextualCompressor:

    def __init__(
        self,
        min_sentence_length=20,
        max_sentences=5
    ):
        self.min_sentence_length = min_sentence_length
        self.max_sentences = max_sentences

    def split_sentences(self, text: str) -> list[str]:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip()
        )

        return [
            sentence.strip()
            for sentence in sentences
            if len(sentence.strip()) >= self.min_sentence_length
        ]

    def compress(
        self,
        query: str,
        documents: list[dict],
        max_sentences=None
    ) -> list[dict]:

        if not documents:
            return []

        query_terms = set(
            re.findall(r"\w+", query.lower())
        )

        sentence_limit = (
            max_sentences
            if max_sentences is not None
            else self.max_sentences
        )

        compressed_documents = []

        for document in documents:

            sentences = self.split_sentences(
                document["content"]
            )

            scored_sentences = []

            for index, sentence in enumerate(sentences):

                sentence_terms = set(
                    re.findall(r"\w+", sentence.lower())
                )

                overlap = len(
                    query_terms.intersection(sentence_terms)
                )

                if overlap > 0:
                    scored_sentences.append(
                        (index, sentence, overlap)
                    )

            # Keep the most relevant sentences.
            selected = sorted(
                scored_sentences,
                key=lambda item: item[2],
                reverse=True
            )[:sentence_limit]

            # Restore original sentence order.
            selected.sort(key=lambda item: item[0])

            compressed_text = " ".join(
                sentence for _, sentence, _ in selected
            )

            if not compressed_text:
                continue

            result = document.copy()

            result["original_content"] = document["content"]
            result["content"] = compressed_text

            result["compression"] = {
                "original_characters": len(document["content"]),
                "compressed_characters": len(compressed_text),
                "sentences_selected": len(selected)
            }

            compressed_documents.append(result)

        return compressed_documents
