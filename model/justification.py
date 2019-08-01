from typing import Optional


documents = {}


class Justification:
    source: str
    start: int
    end: int
    label: Optional[str]
    type_: Optional[str]

    def __init__(self, source, start, end, label, type_):
        try:
            self.source = documents[source]
        except KeyError:
            print("Source is not defined: ", source)
        self.start = start
        self.end = end
        self.label = label
        self.type = type_

    def __str__(self):
        if not self.label:
            return '<J:  {}-{}>'.format(self.start, self.end)
        return '<J: {}-{} "{}">'.format(self.start, self.end, self.label)

    def __repr__(self):
        return self.__str__()


class Document:
    def __init__(self, doc_id, lang):
        self.id = doc_id
        self.lang = lang
        documents[self.id] = self

    def __str__(self):
        return '<{}: {}>'.format(self.id, self.lang)

    def __repr__(self):
        return self.__str__()
