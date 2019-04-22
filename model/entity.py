from model.justification import Justification
from collections import defaultdict, Counter
from typing import List


abc: List[int] = '123'

elements = {}
entities = {}
events = {}
relations = {}
targets = defaultdict(set)


class BaseElement:
    uri: str
    type: str
    justifications: List[Justification]

    def __init__(self, uri, type_):
        self.uri = uri
        self.type = type_
        self.justifications = []
        elements[uri] = self

    def add_justification(self, source, start, end, label, type_):
        self.justifications.append(Justification(source, start, end, label, type_))

    @property
    def source(self):
        return self.justifications[0].source if self.justifications else None

    @property
    def all_labels(self):
        return [j.label for j in self.justifications if j.label]

    def __eq__(self, other):
        return isinstance(other, BaseElement) and self.uri == other.uri

    def __hash__(self):
        return hash(self.uri)


class Entity(BaseElement):
    target: str

    def __init__(self, uri, type_, target):
        super().__init__(uri, type_)
        self.target = target
        entities[uri] = self
        if target:
            targets[target].add(self)

    @property
    def mention_labels(self):
        return [j.label for j in self.justifications if j.label and j.type == 'mention']

    @property
    def nominal_labels(self):
        return [j.label for j in self.justifications if j.label and j.type == 'nominal_mention']

    @property
    def pronominal_labels(self):
        return [j.label for j in self.justifications if j.label and j.type == 'pronominal_mention']

    def __str__(self):
        labels = self.mention_labels
        if labels:
            label = Counter(labels).most_common(1)[0]
            return '<[E-{}]: {}>'.format(self.type, label)
        else:
            return '<[E-{}]>'.format(self.type)

    def __repr__(self):
        return self.__str__()


class Event(BaseElement):
    def __init__(self, uri, type_):
        super().__init__(uri, type_)
        events[uri] = self
        self.roles = defaultdict(set)

    def add_role(self, role, ent):
        self.roles[role].add(ent)


class Relation(BaseElement):
    def __init__(self, uri, type_):
        super().__init__(uri, type_)
        relations[uri] = self
