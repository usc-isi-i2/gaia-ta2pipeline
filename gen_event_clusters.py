from gastrodon import RemoteEndpoint, inline
import json

namespaces_str = """
@prefix : <https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/AidaDomainOntologiesCommon#> .
@prefix aida: <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/InterchangeOntology#> .
@prefix dc: <http://purl.org/dc/elements/1.1/> .
@prefix domainOntology: <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/SeedlingOntology> .
@prefix ldc: <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LdcAnnotations#> .
@prefix ldcOnt: <https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LDCOntology#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def gen_event_clusters(endpoint_url, outdir):

    endpoint = RemoteEndpoint(url=endpoint_url,
                              prefixes=inline(namespaces_str).graph)

    df = endpoint.select("""
    SELECT DISTINCT ?e {
        SELECT ?e 
        WHERE {
            ?e a aida:Event;
        }
    }
    """)

    def to_list(e):
        return [str(e)]

    events = df['e'].apply(to_list).values.tolist()

    with open(outdir + '/event-clusters.jl', 'w') as f:
        for c in events:
            f.write(json.dumps(c) + '\n')
