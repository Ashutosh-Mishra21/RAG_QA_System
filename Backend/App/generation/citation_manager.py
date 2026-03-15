import re


class CitationManager:

    def extract(self, answer):

        citations = re.findall(r"\[(\d+)\]", answer)

        return list(set(citations))
