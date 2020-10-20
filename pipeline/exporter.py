import pandas as pd
import glob
import os
import sys
import re
from collections import defaultdict
from config import config, get_logger


# run validator
# docker run --rm -it -v /tmp/aif_validator:/v -e VALIDATION_HOME=/opt/aif-validator -e VALIDATION_FLAGS=--ldc -e TARGET_TO_VALIDATE=/v --name aifvalidator nextcenturycorp/aif_validator


logger = get_logger('exporter')

NAMESPACE = {
    'gaia': 'http://www.isi.edu/gaia/',
    'xsd': 'http://www.w3.org/2001/XMLSchema#'
}

ENTITY_TEMPLATE = """{} a       aida:Entity ;
        aida:system  gaia:TA1 .\n"""

CLUSTER_TEMPLATE = """{}  a        aida:SameAsCluster ;
        aida:prototype  {} ;
        aida:system     gaia:TA2 .\n"""

ENTITY_ASSERTION_TEMPLATE = """{}  a        rdf:Statement ;
        rdf:object        ldcOnt:PER ;
        rdf:predicate     rdf:type ;
        rdf:subject       {} ;
        aida:confidence   [ a                     aida:Confidence ;
                            aida:confidenceValue  "1.0"^^xsd:double ;
                            aida:system           gaia:TA2
                          ] ;
        aida:justifiedBy  [ a                        aida:TextJustification ;
                            aida:confidence          [ a                     aida:Confidence ;
                                                       aida:confidenceValue  "1.0"^^xsd:double ;
                                                       aida:system           gaia:TA2
                                                     ] ;
                            aida:source              "{}" ;
                            aida:sourceDocument      "{}" ;
                            aida:startOffset         "0"^^xsd:int ;
                            aida:endOffsetInclusive  "0"^^xsd:int ;
                            aida:system              gaia:TA2
                          ] ;
        aida:system       gaia:TA2 .\n"""

MEMBERSHIP_TEMPLATE = """[ a                   aida:ClusterMembership ;
  aida:cluster        {} ;
  aida:clusterMember  {} ;
  aida:confidence     [ a                     aida:Confidence ;
                        aida:confidenceValue  "{}"^^xsd:double ;
                        aida:system           gaia:TA2
                      ] ;
  aida:system         gaia:TA2
] .\n"""

COLUMNS = ['e', 'name', 'type', 'target', 'target_score', 'target_type',
           'target_name', 'fbid', 'fbid_score_avg', 'fbid_score_max', 'wikidata',
           'wikidata_label_en', 'wikidata_label_ru', 'wikidata_label_uk',
           'wikidata_description_en', 'wikidata_description_ru',
           'wikidata_description_uk', 'wikidata_alias_en', 'wikidata_alias_ru',
           'wikidata_alias_uk', 'infojust_confidence', 'informative_justification',
           'just_confidence', 'justified_by', 'source', 'cluster', 'synthetic', 'cluster_member_confidence']

ESEENTIAL_COLUMNS = ["e",
                     "cluster",
                     "infojust_confidence",
                     "informative_justification",
                     "just_confidence",
                     "justified_by",
                     "source",
                     "cluster_member_confidence"]


class Exporter(object):
    def __init__(self, infile, outfile):

        df = pd.read_hdf(infile)
        self.fp = open(outfile, "w")
        self.df = df[df["synthetic"] == False][ESEENTIAL_COLUMNS]
        self.proto_df = df[df["synthetic"] == True][ESEENTIAL_COLUMNS]
        self.n = self.df.shape[0]
        self.entities = None
        self.clusters = set()
        self.ns_mapping = self.__class__.generate_name_space()

    @classmethod
    def generate_name_space(cls):
        '''
        a help function to generate a mapping from prefix to full url
        '''
        namespace_df = pd.read_csv(config['namespace_file'], sep="\t")
        name_space = {}
        for i in range(namespace_df.shape[0]):
            record = namespace_df.iloc[i]
            name_space[record["node1"]] = record["node2"]

        # additional namespace
        for k, v in NAMESPACE.items():
            if k not in name_space:
                name_space[k] = v
        return name_space

    def extend_prefix(self, s):
        ss = s.split(':')
        if len(ss) > 1:
            p = ss[0]
            s = self.ns_mapping.get(p, p) + ':'.join(ss[1:])
        return '<{}>'.format(s)

    def run(self):
        self.declare_prefix()
        # self.declare_entity()
        self.declare_cluster()
        self.declare_system()
        self.declare_entity_cluster_membership()
        # self.declare_entity_assertion()

    def __dell__(self):
        self.fp.close()

    def declare_prefix(self):
        # prefix_dict = self.__class__.generate_name_space()
        prefix_dict = self.ns_mapping
        prefix_list = ["@prefix {}: <{}> .".format(pre, url) for
                       pre, url in prefix_dict.items()]
        self.write("\n".join(prefix_list) + "\n")

    def write(self, string: str):
        self.fp.write(string)
        self.fp.write("\n")

    def declare_entity(self):
        self.entities = self.df["e"].to_list()
        self.entity_sources = self.df["source"].to_list()
        # for entity in self.entities:
        #     # TODO bypass columbia illegal delcaration
        #     if self.__class__.legal_filter(entity):
        #         entity = self.extend_prefix(entity)
        #         entity_statement = ENTITY_TEMPLATE.format(entity)
        #         self.write(entity_statement)

    @classmethod
    def legal_filter(cls, *strings):
        # a filter to filter illegal entities
        for s in strings:
            if s.startswith("columbia"):
                return False
        return True

    def declare_cluster(self):
        protos = self.proto_df["e"].to_list()
        clusters = self.proto_df["cluster"].to_list()
        for proto, cluster in zip(protos, clusters):
            # TODO
            # by pass columbia illegal declaration
            assert (type(cluster) == tuple and len(cluster) == 1)
            cluster = cluster[0]
            if cluster not in self.clusters and self.__class__.legal_filter(cluster):
                self.clusters.add(cluster)
                cluster = self.extend_prefix(cluster)
                proto = self.extend_prefix(proto)
                cluster_info = CLUSTER_TEMPLATE.format(cluster,
                                                       proto)
                self.write(cluster_info)

    def declare_system(self):
        # manual
        # ta1 = "gaia:TA1  a  aida:System .\n"
        ta2 = "gaia:TA2  a  aida:System .\n"
        # self.write(ta1)
        self.write(ta2)

    def declare_entity_cluster_membership(self):
        '''
        use the prototype entities to declare cluster membership
        '''
        entities = self.df["e"].to_list()
        clusters = self.df["cluster"].to_list()
        confidences = self.df["cluster_member_confidence"].to_list()
        for entity, cluster, confidence in zip(entities, clusters, confidences):
            entity = self.extend_prefix(entity)
            # TODO handle possilbe types of cluster: string or tuple
            if type(cluster) == tuple:
                for idx, cluster_ in enumerate(cluster):
                    cluster_ = self.extend_prefix(cluster_)
                    if self.__class__.legal_filter(cluster_, entity):
                        membership_info = MEMBERSHIP_TEMPLATE.format(cluster_, entity, confidence[idx])
                        self.write(membership_info)
            else:
                cluster_ = cluster
                cluster_ = self.extend_prefix(cluster_)
                if self.__class__.legal_filter(cluster_, entity):
                    membership_info = MEMBERSHIP_TEMPLATE.format(cluster_, entity, confidence)
                    self.write(membership_info)

    def declare_entity_assertion(self):
        # use entity id for assertion id
        for entity, source in zip(self.entities, self.entity_sources):
            if self.__class__.legal_filter(entity, source):
                # a mimic of other objects like `entity`, `relations`
                assertion = "assertion:" + ':'.join(entity.split(':')[1:])
                assertion = self.extend_prefix(assertion)
                entity = self.extend_prefix(entity)
                assertion_info = ENTITY_ASSERTION_TEMPLATE.format(assertion,
                                                                  entity,
                                                                  source,
                                                                  source)
                self.write(assertion_info)


def process():

    logger.info('exporting entity clusters')

    output_dir = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(output_dir, exist_ok=True)

    for infile in glob.glob(os.path.join(config['temp_dir'], config['run_name'], 'entity_cluster.h5')):
        outfile = os.path.join(output_dir, 'ta2_entity_cluster.ttl')
        exporter = Exporter(infile, outfile)
        exporter.run()


    # # assign bnode globally unique id
    # counter = [0]
    # re_bnode = re.compile(r'_:([a-z0-9]+)')
    # ta1_concatenated_file = os.path.join(config['temp_dir'], config['run_name'], 'ta1_concatenated.nt')
    # for infile in glob.glob(os.path.join(config['input_dir'], config['run_name'], '*.ttl.nt')):
    #     source = os.path.basename(infile).split('.')[0]
    #
    #     bnode_mapping = {}
    #
    #     def replace_bnode(bnode, counter):
    #         if bnode not in bnode_mapping:
    #             bnode_mapping[bnode] = counter[0]
    #             counter[0] += 1
    #         return '_:b{}'.format(bnode_mapping[bnode])
    #
    #     fout = open(ta1_concatenated_file, 'w')
    #     with open(infile, 'r') as fin:
    #         for idx, line in enumerate(fin):
    #             fout.write(re_bnode.sub(lambda x: replace_bnode(x.group(), counter), line))
    #     fout.close()


if __name__ == '__main__':
    argv = sys.argv
    if argv[1] == 'process':
        process()
