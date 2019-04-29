from gastrodon import RemoteEndpoint,QName,ttl,URIRef,inline
import pandas as pd
import json
from gastrodon import _parseQuery
from SPARQLWrapper import SPARQLWrapper, N3
from rdflib import Graph
from model.source import LTFSourceContext
from rdflib.plugins.stores.sparqlstore import SPARQLStore
from rdflib.namespace import Namespace, RDFS, SKOS
from rdflib import URIRef, Literal


# wikidata_sparql = SPARQLStore("http://sitaware.isi.edu:8080/bigdata/namespace/wdq/sparql")
wikidata_sparql = SPARQLStore("https://query.wikidata.org/sparql")
WDT = Namespace('http://www.wikidata.org/prop/direct/')
namespaces = {'wdt': WDT, 'skos': SKOS}

namespaces_str = """
@prefix : <https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/AidaDomainOntologiesCommon#> .
@prefix aida: <https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/InterchangeOntology#> .
@prefix dc: <http://purl.org/dc/elements/1.1/> .
@prefix domainOntology: <https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/SeedlingOntology> .
@prefix ldc: <https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/LdcAnnotations#> .
@prefix ldcOnt: <https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/LDCOntology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def query_context(source, start, end):
    if start == -1 or end == -1:
        return None
    context_extractor = LTFSourceContext(source)
    if context_extractor.doc_exists():
        return context_extractor.query_context(start, end)


def link_wikidata(fbid):
    if not fbid or fbid.startswith('LDC2015E42:NIL'):
        return None
    fbid = '/' + fbid[11:].replace('.', '/')
    query = "SELECT ?qid WHERE { ?qid wdt:P646 ?freebase } LIMIT 1"
    for qid, in wikidata_sparql.query(query, namespaces, {'freebase': Literal(fbid)}):
        return str(qid)


def get_labels(pred, lang):
    def get_labels_for_entity(entity):
        if not entity:
            return None
        query = """
        SELECT ?label 
        WHERE { 
            ?qid pred ?label
            FILTER (lang(?label) = "language") }
        """.replace('pred', pred).replace('language', lang)
        labels = []
        for label, in wikidata_sparql.query(query, namespaces, {'qid': URIRef(entity)}):
            labels.append(str(label))
        return tuple(labels)

    return get_labels_for_entity


def to_int(s):
    return int(s) if isinstance(s, str) or isinstance(s, int) else -1


def generate_dataframe(endpoint_url, outdir):
    endpoint = RemoteEndpoint(url=endpoint_url, prefixes=inline(namespaces_str).graph)

    def describe(self, sparql: str):
        return self._describe(sparql).serialize(format='n3').decode()

    def _describe(self, sparql: str):
        that = endpoint._wrapper()
        that.setQuery(endpoint._prepend_namespaces(sparql, _parseQuery))
        that.setReturnFormat(N3)
        results = that.query().convert()
        g = Graph()
        g.parse(data=results, format="n3")
        return g

    RemoteEndpoint.describe = describe
    RemoteEndpoint._describe = _describe

    df = endpoint.select("""
    SELECT DISTINCT ?e ?fbid {
        ?e a aida:Entity ;
           aida:system <http://www.rpi.edu> ;
           ^rdf:subject/aida:justifiedBy/aida:privateData [
                aida:jsonContent ?fbid ;
                aida:system <http://www.rpi.edu/EDL_Freebase>
            ]
    }
    """)
    df.fbid = df.fbid.apply(lambda s: json.loads(s).get('freebase_link') if s else None)
    df = df.astype({
        'e': str, 'fbid': str
    })
    rpi_external = df

    df = endpoint.select("""
    SELECT DISTINCT ?e ?type ?label ?target ?source ?start ?end ?justificationType {
        ?e a aida:Entity ;
           aida:system <http://www.rpi.edu> ;
           ^rdf:subject [
            a rdf:Statement ;
            rdf:predicate rdf:type ;
            rdf:object ?type ;
            aida:justifiedBy ?justification ]
        OPTIONAL { ?justification aida:privateData [
                aida:jsonContent ?label ;
                aida:system <http://www.rpi.edu/EDL_Translation> ]}
        OPTIONAL { ?e aida:link/aida:linkTarget ?target }
        OPTIONAL { ?justification aida:source ?source }
        OPTIONAL { ?justification aida:startOffset ?start }
        OPTIONAL { ?justification aida:endOffsetInclusive ?end }
        OPTIONAL { ?justification aida:privateData [ 
                aida:system <http://www.rpi.edu> ;
                aida:jsonContent ?justificationType ] }
    }
    """)
    df.start = df.start.apply(to_int)
    df.end = df.end.apply(to_int)
    df.justificationType = df.justificationType.apply(lambda s: json.loads(s).get('justificationType'))
    df.label = df.label.apply(lambda s: tuple(json.loads(s).get('translation')) if s else None)
    df = df.astype({
        'e': str, 'type': str, 'target': str, 'source': str, 'start': int, 'end': int, 'justificationType': str
    })
    rpi_entity_with_justification = df

    df = endpoint.select("""
    SELECT DISTINCT ?e ?type ?name ?text ?target ?source {
        ?e a aida:Entity ;
           aida:justifiedBy/aida:source ?source ;
           aida:system <http://www.rpi.edu> .
        ?statement a rdf:Statement ;
                   rdf:subject ?e ;
                   rdf:predicate rdf:type ;
                   rdf:object ?type .
        OPTIONAL { ?e aida:hasName ?name }
        OPTIONAL { ?e aida:textValue ?text }
        OPTIONAL { ?e aida:link/aida:linkTarget ?target }
    }
    """)
    df = df.astype({
        'e': str, 'type': str, 'name': str, 'target': str, 'source': str, 'text': str
    })
    rpi_entity_valid = df

    # Relations

    df = endpoint.select("""
    SELECT DISTINCT ?e ?type ?source ?start ?end {
        ?e a aida:Relation ;
           aida:system <http://www.rpi.edu> .
        ?statement a rdf:Statement ;
                   rdf:subject ?e ;
                   rdf:predicate rdf:type ;
                   rdf:object ?type ;
                   aida:justifiedBy ?justification 
        OPTIONAL { ?justification aida:source ?source }
        OPTIONAL { ?justification aida:startOffset ?start }
        OPTIONAL { ?justification aida:endOffsetInclusive ?end }
    }
    """)
    df.start = df.start.apply(to_int)
    df.end = df.end.apply(to_int)
    df = df.astype({
        'e': str, 'type': str, 'source': str, 'start': int, 'end': int
    })
    rpi_relation = df

    df = endpoint.select("""
    SELECT DISTINCT ?e ?p ?o {
        ?e a aida:Relation ;
           aida:system <http://www.rpi.edu> .
        ?statement a rdf:Statement ;
                   rdf:subject ?e ;
                   rdf:predicate ?p ;
                   rdf:object ?o 
        FILTER (?p != rdf:type)
    }
    """)
    df = df.astype({
        'e': str, 'p': str, 'o': str
    })
    rpi_relation_roles = df

    # Documents

    df = endpoint.select("""
    SELECT DISTINCT ?source ?fileType {
        ?justification a aida:TextJustification ;
                       aida:system <http://www.rpi.edu> ;
                       aida:source ?source ;
                       aida:privateData ?filePrivate .
        ?filePrivate aida:system <http://www.rpi.edu/fileType> ;
                     aida:jsonContent ?fileType
    }
    """)
    df['lang'] = df.fileType.apply(lambda s: json.loads(s).get('fileType'))
    df = df.drop(columns='fileType')
    df = df.astype({
        'source': str, 'lang': str
    })
    document_types = df

    rpi_entity_with_justification.to_hdf(outdir + '/entity_with_labels.h5', 'entity', mode='w', format='fixed')
    rpi_entity_valid.to_hdf(outdir + '/entity_valid.h5', 'entity', mode='w', format='fixed')
    rpi_relation.to_hdf(outdir + '/relation.h5', 'entity', mode='w', format='fixed')
    rpi_relation_roles.to_hdf(outdir + '/relation_roles.h5', 'entity', mode='w', format='fixed')
    document_types.to_hdf(outdir + '/document.h5', 'entity', mode='w', format='fixed')
    _ = pd.read_hdf(outdir + '/entity_with_labels.h5')
    _ = pd.read_hdf(outdir + '/entity_valid.h5')
    _ = pd.read_hdf(outdir + '/relation.h5')
    _ = pd.read_hdf(outdir + '/relation_roles.h5')
    _ = pd.read_hdf(outdir + '/document.h5')

    # Transform Entities

    # 1. `name`

    df = rpi_entity_valid
    df['name'] = df.apply(lambda r: r['name'] if r['name'] != 'None' else None, axis=1)
    df = df.drop(columns='text')
    df = df.drop_duplicates()
    df = df[['e', 'type', 'target', 'source']].groupby('e').head(1).join(df.groupby('e')['name'].apply(tuple), on='e')
    df['name'] = df['name'].apply(lambda s: s if s[0] else None)
    df_names = df
    df.head()

    # 2. origin

    rpi_entity_with_justification['origin'] = rpi_entity_with_justification.apply(lambda r: query_context(r.source, r.start, r.end), axis=1)
    rpi_entity_with_justification.head()

    df = rpi_entity_with_justification
    # drop entities with nominal mention and pronominal mention
    # comment out line below to generate for all entities
    # df = df[(df['justificationType']!='nominal_mention') & (df['justificationType']!='pronominal_mention')]
    df['debug'] = df['justificationType'].apply(
        lambda s: False if s != 'nominal_mention' and s != 'pronominal_mention' else True)

    rpi_entity_with_justification_filtered = df
    df_origin = df[['e', 'origin']].groupby('e')['origin'].apply(tuple).to_frame()
    df_origin['origin'] = df_origin['origin'].apply(lambda s: s if s[0] else None)
    df_origin.head()

    # 3. wikidata and wikidata labels, alias

    df_target = rpi_external[['fbid']].drop_duplicates()
    df_target['wikidata'] = df_target.fbid.apply(link_wikidata)
    df_target['wiki_label_en'] = df_target['wikidata'].apply(get_labels('rdfs:label', 'en'))
    df_target['wiki_label_ru'] = df_target['wikidata'].apply(get_labels('rdfs:label', 'ru'))
    df_target['wiki_label_uk'] = df_target['wikidata'].apply(get_labels('rdfs:label', 'uk'))
    df_target['wiki_alias_en'] = df_target['wikidata'].apply(get_labels('skos:altLabel', 'en'))
    df_target['wiki_alias_ru'] = df_target['wikidata'].apply(get_labels('skos:altLabel', 'ru'))
    df_target['wiki_alias_uk'] = df_target['wikidata'].apply(get_labels('skos:altLabel', 'uk'))
    df_target['fbid_type'] = df_target.fbid.apply(lambda t: 'm' if ':m' in t else 'NIL')

    df = rpi_entity_with_justification_filtered[['e', 'type', 'label', 'source', 'target', 'debug']].drop_duplicates().join(df_origin, on='e')
    df = df.join(rpi_external.set_index('e'), on='e')
    df = df.join(df_target.set_index('fbid'), on='fbid')
    df = df.join(document_types.set_index('source'), on='source')
    df = df.join(df_names[['e', 'name']].set_index('e'), on='e')
    df = df[['e', 'type', 'name', 'source', 'target', 'fbid', 'fbid_type', 'wikidata',
           'wiki_label_en', 'wiki_label_ru', 'wiki_label_uk', 'wiki_alias_en',
           'wiki_alias_ru', 'wiki_alias_uk', 'origin', 'lang', 'label', 'debug']]
    df_all = df

    df_all.to_hdf(outdir + '/entity_all.h5', 'entity', mode='w', format='fixed')
    _ = pd.read_hdf(outdir + '/entity_all.h5')

    df_all.to_csv(outdir + '/entity_all.csv')


