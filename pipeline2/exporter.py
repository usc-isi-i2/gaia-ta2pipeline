import pandas as pd
import glob
import os
import sys
import re
from collections import defaultdict
from config import config, get_logger
from common import exec_sh


logger = get_logger('exporter')

NAMESPACE = {
    'gaia': 'http://www.isi.edu/gaia/',
    'xsd': 'http://www.w3.org/2001/XMLSchema#'
}

# ENTITY_TEMPLATE = """{} a       aida:Entity ;
#         aida:system  gaia:TA1 .\n"""

SYSTEM_TEMPLATE = """
gaia:TA2    a   aida:System .\n"""

CLUSTER_TEMPLATE = """{cluster}  a        aida:SameAsCluster ;
        aida:prototype  {proto} ;
        aida:system     gaia:TA2 .\n"""

PROTOTYPE_TEMPLATE = """{proto} a aida:Entity ;
    aida:confidence [ a aida:Confidence ;
            aida:confidenceValue "{cv}"^^xsd:double ;
            aida:system gaia:TA2 ] ;
{info_just}
{link}
    aida:system gaia:TA2 .\n"""

PROTOTYPE_INFOJUST_TEMPLATE = """    aida:informativeJustification {info_just} ;\n"""

PROTOTYPE_LINK_TEMPLATE = """
    aida:link [ a aida:LinkAssertion ;
        aida:confidence [ a aida:Confidence ;
            aida:confidenceValue "{link_cv}"^^xsd:double ;
            aida:system gaia:TA2 ] ;
        aida:linkTarget "{link}"^^xsd:string ;
        aida:system gaia:TA2 ] ;\n"""


PROTOTYPE_TYPE_TEMPLATE = """[ a rdf:Statement;
        rdf:object        "{type_}"^^xsd:string ;
        rdf:predicate     rdf:type ;
        rdf:subject       {proto} ;
        aida:confidence   [ a   aida:Confidence ;
            aida:confidenceValue  "{cv}"^^xsd:double ;
            aida:system           gaia:TA2
        ] ;
{just}
        aida:system       gaia:TA2
] .\n"""

PROTOTYPE_TYPE_JUST_TEMPLATE = """        aida:justifiedBy {just} ;\n"""

# ENTITY_ASSERTION_TEMPLATE = """{}  a        rdf:Statement ;
#         rdf:object        ldcOnt:PER ;
#         rdf:predicate     rdf:type ;
#         rdf:subject       {} ;
#         aida:confidence   [ a                     aida:Confidence ;
#                             aida:confidenceValue  "1.0"^^xsd:double ;
#                             aida:system           gaia:TA2
#                           ] ;
#         aida:justifiedBy  [ a                        aida:TextJustification ;
#                             aida:confidence          [ a                     aida:Confidence ;
#                                                        aida:confidenceValue  "1.0"^^xsd:double ;
#                                                        aida:system           gaia:TA2
#                                                      ] ;
#                             aida:source              "{}" ;
#                             aida:sourceDocument      "{}" ;
#                             aida:startOffset         "0"^^xsd:int ;
#                             aida:endOffsetInclusive  "0"^^xsd:int ;
#                             aida:system              gaia:TA2
#                           ] ;
#         aida:system       gaia:TA2 .\n"""

MEMBERSHIP_TEMPLATE = """[ a                   aida:ClusterMembership ;
  aida:cluster        {cluster} ;
  aida:clusterMember  {member} ;
  aida:confidence     [ a                     aida:Confidence ;
                        aida:confidenceValue  "{cv}"^^xsd:double ;
                        aida:system           gaia:TA2
                      ] ;
  aida:system         gaia:TA2
] .\n"""

ASSO_CLAIM_TEMPLATE = """{claim} aida:associatedKEs {cluster} .\n"""
CLAIM_SEMAN_TEMPLATE = """{claim} aida:claimSemantics {cluster} .\n"""


# SUPER_EDGE_TEMPLATE = """[ a rdf:Statement ;
#     rdf:subject         {proto1} ;
#     rdf:predicate       {edge_type} ;
#     rdf:object          {proto2} ;
#     aida:justifiedBy    {just} ;
#     aida:system         gaia:TA2
# ] .\n"""

# SUPER_EDGE_COMPOUND_JUSTIFICATION = """aida:containedJustification {infojust} ;\n"""

# COLUMNS = ['e', 'name', 'type', 'target', 'target_score', 'target_type',
#            'target_name', 'fbid', 'fbid_score_avg', 'fbid_score_max', 'wikidata',
#            'wikidata_label_en', 'wikidata_label_ru', 'wikidata_label_uk',
#            'wikidata_description_en', 'wikidata_description_ru',
#            'wikidata_description_uk', 'wikidata_alias_en', 'wikidata_alias_ru',
#            'wikidata_alias_uk', 'infojust_confidence', 'informative_justification',
#            'just_confidence', 'justified_by', 'source', 'cluster', 'synthetic', 'cluster_member_confidence']
#
# ESSENTIAL_COLUMNS = ["e",
#                      "cluster",
#                      # "infojust_confidence",
#                      # "informative_justification",
#                      # "just_confidence",
#                      # "justified_by",
#                      "source",
#                      "cluster_member_confidence"]
#
# ESSENTIAL_COLUMNS_RELATION = [
#     'prototype1', 'prototype2', 'role', 'importance', 'infojust', 'compound_cv'
# ]


class Exporter(object):
    def __init__(self, entity, event, event_role, relation, relation_role, outfile):

        df = pd.read_hdf(entity)
        self.fp = open(outfile, "w")
        self.df = df[df["synthetic"] == False] # [ESSENTIAL_COLUMNS]
        self.proto_df = df[df["synthetic"] == True] # [ESSENTIAL_COLUMNS]
        self.df_event = None  # pd.read_hdf(event)
        self.df_event_role = None  # pd.read_hdf(event_role)
        self.df_relation = None  # pd.read_hdf(relation)
        self.df_relation_role = None  # pd.read_hdf(relation_role)
        self.n = self.df.shape[0]
        self.entities = None
        self.clusters = set()
        self.ns_mapping = self.__class__.generate_name_space()

        logger.info(f'# of clusters: {len(set(self.df["cluster"].to_list()))}')
        logger.info(f'# of prototypes: {len(self.proto_df)}')

        # print(self.df[self.df['asso_claim'].notnull()][['asso_claim', 'cluster']])
        # print(self.df[self.df['claim_seman'].notnull()][['claim_seman', 'cluster']])

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
        if s.startswith('<') and s.endswith('>'):
            return s
        if not s.startswith(('http://', 'https://')):
            ss = s.split(':')
            if len(ss) > 1:
                p = ss[0]
                s = self.ns_mapping.get(p, p) + ':'.join(ss[1:])
        return '<{}>'.format(s)

    def run(self):
        self.declare_prefix()
        # self.declare_entity()
        self.declare_cluster()
        self.declare_cluster_membership()
        self.declare_prototype()
        # self.declare_entity_assertion()
        # self.declare_super_edge()
        self.declare_claims()
        self.fp.flush()  # file could be truncated without this line

    def __dell__(self):
        self.fp.close()

    def declare_prefix(self):
        # prefix_dict = self.__class__.generate_name_space()
        prefix_dict = self.ns_mapping
        prefix_list = ["@prefix {}: <{}> .".format(pre, url) for
                       pre, url in prefix_dict.items()]
        self.write("\n".join(prefix_list) + "\n")

        self.write(SYSTEM_TEMPLATE)

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
            # # TODO
            # # by pass columbia illegal declaration
            # assert (type(cluster) == tuple and len(cluster) == 1)
            # cluster = cluster[0]
            # if cluster not in self.clusters and self.__class__.legal_filter(cluster):
            if cluster not in self.clusters:
                self.clusters.add(cluster)
                cluster = self.extend_prefix(cluster)
                proto = self.extend_prefix(proto)
                cluster_info = CLUSTER_TEMPLATE.format(cluster=cluster, proto=proto)
                self.write(cluster_info)

        # # event
        # for e in self.df_event['e'].to_list():
        #     cluster = e + '-cluster'
        #     proto = e
        #     cluster = self.extend_prefix(cluster)
        #     proto = self.extend_prefix(proto)
        #     cluster_info = CLUSTER_TEMPLATE.format(cluster=cluster, proto=proto)
        #     self.write(cluster_info)
        # # relation
        # for e in self.df_relation['e'].to_list():
        #     cluster = e + '-cluster'
        #     proto = e
        #     cluster = self.extend_prefix(cluster)
        #     proto = self.extend_prefix(proto)
        #     cluster_info = CLUSTER_TEMPLATE.format(cluster=cluster, proto=proto)
        #     self.write(cluster_info)

    def declare_cluster_membership(self):
        '''
        use the prototype entities to declare cluster membership
        '''
        entities = self.df["e"].to_list()
        clusters = self.df["cluster"].to_list()
        confidences = self.df["cluster_member_cv"].to_list()
        for entity, cluster, confidence in zip(entities, clusters, confidences):
            entity = self.extend_prefix(entity)
            # TODO handle possilbe types of cluster: string or tuple
            if type(cluster) == tuple:
                for idx, cluster_ in enumerate(cluster):
                    cluster_ = self.extend_prefix(cluster_)
                    if self.__class__.legal_filter(cluster_, entity):
                        membership_info = MEMBERSHIP_TEMPLATE.format(cluster=cluster_, member=entity, cv=confidence[idx])
                        self.write(membership_info)
            else:
                cluster_ = cluster
                cluster_ = self.extend_prefix(cluster_)
                if self.__class__.legal_filter(cluster_, entity):
                    membership_info = MEMBERSHIP_TEMPLATE.format(cluster=cluster_, member=entity, cv=confidence)
                    self.write(membership_info)

        # # event
        # for e in self.df_event['e'].to_list():
        #     cluster = e + '-cluster'
        #     member = e
        #     cluster = self.extend_prefix(cluster)
        #     member = self.extend_prefix(member)
        #     membership_info = MEMBERSHIP_TEMPLATE.format(cluster=cluster, member=member, cv=1.0)
        #     self.write(membership_info)
        # # relation
        # for e in self.df_relation['e'].to_list():
        #     cluster = e + '-cluster'
        #     member = e
        #     cluster = self.extend_prefix(cluster)
        #     member = self.extend_prefix(member)
        #     membership_info = MEMBERSHIP_TEMPLATE.format(cluster=cluster, member=member, cv=1.0)
        #     self.write(membership_info)

    def declare_prototype(self):
        for idx, row in self.proto_df.iterrows():
            proto = row['e']
            cluster = row['cluster']

            # select one info_just for each source document (nist)
            info_justs = self.df[self.df['cluster'] == cluster][['info_just', 'source']]
            info_justs = info_justs.groupby('source').head(1)['info_just']
            info_justs = list(set(info_justs.to_list()))

            links = self.proto_df['link'].to_list()[0]
            link_cvs = self.proto_df['link_cv'].to_list()[0]

            proto = self.extend_prefix(proto)
            cv = 1.0

            link_str = ''
            for link, cv in zip(links, link_cvs):
                link_str += PROTOTYPE_LINK_TEMPLATE.format(link=link, link_cv=cv)

            ij_str = ''
            # if len(info_justs) > 1:
                # print(info_justs)
            for ij in info_justs:
                ij = self.extend_prefix(ij)
                ij_str += PROTOTYPE_INFOJUST_TEMPLATE.format(info_just=ij)

            proto_info = PROTOTYPE_TEMPLATE.format(proto=proto, cv=cv, info_just=ij_str, link=link_str)
            self.write(proto_info)

            # type and type justification
            types = self.proto_df['type'].to_list()[0]
            type_cvs = self.proto_df['type_cv'].to_list()[0]
            type_just = self.df[self.df['cluster'] == cluster][['type', 'type_just']]
            t_to_j_mapping = defaultdict(set)  # type to justification mapping, aggregate all justifications
            for _, row in type_just.iterrows():
                ts = row['type']
                tjs = row['type_just']
                for t, j in zip(ts, tjs):
                    for jj in j:
                        t_to_j_mapping[t].add(jj)

            # create statement for each type
            for t in t_to_j_mapping:
                type_just_str = ''
                for just in t_to_j_mapping[t]:
                    just = self.extend_prefix(just)
                    type_just_str += PROTOTYPE_TYPE_JUST_TEMPLATE.format(just=just)

                type_str = ''
                for type_, cv in zip(types, type_cvs):
                    type_str += PROTOTYPE_TYPE_TEMPLATE.format(proto=proto, type_=type_, cv=cv, just=type_just_str)
                self.write(type_str)

    def declare_claims(self):
        for idx, row in self.proto_df.iterrows():
            cluster = row['cluster']

            dedup_asso_claim = set()
            asso_claim = self.df[self.df['cluster'] == cluster]['asso_claim'].to_list()
            for claims in asso_claim:
                if pd.isna(claims):
                    continue
                for claim in claims:
                    dedup_asso_claim.add(claim)
            asso_claim = list(dedup_asso_claim)

            for claim in asso_claim:
                claim_info = ASSO_CLAIM_TEMPLATE.format(claim=self.extend_prefix(claim), cluster=self.extend_prefix(cluster))
                self.write(claim_info)

            dedup_claim_seman = set()
            claim_seman = self.df[self.df['cluster'] == cluster]['claim_seman'].to_list()
            for claims in claim_seman:
                if pd.isna(claims):
                    continue
                for claim in claims:
                    dedup_claim_seman.add(claim)
            claim_seman = list(dedup_claim_seman)

            for claim in claim_seman:
                claim = self.extend_prefix(claim)
                cluster = self.extend_prefix(cluster)
                claim_info = CLAIM_SEMAN_TEMPLATE.format(claim=self.extend_prefix(claim), cluster=self.extend_prefix(cluster))
                self.write(claim_info)


    # def declare_entity_assertion(self):
    #     # use entity id for assertion id
    #     for entity, source in zip(self.entities, self.entity_sources):
    #         if self.__class__.legal_filter(entity, source):
    #             # a mimic of other objects like `entity`, `relations`
    #             assertion = "assertion:" + ':'.join(entity.split(':')[1:])
    #             assertion = self.extend_prefix(assertion)
    #             entity = self.extend_prefix(entity)
    #             assertion_info = ENTITY_ASSERTION_TEMPLATE.format(assertion,
    #                                                               entity,
    #                                                               source,
    #                                                               source)
    #             self.write(assertion_info)
    #
    # def declare_super_edge(self):
    #     for idx, row in self.df_relation_role.iterrows():
    #         proto1 = self.extend_prefix(row['prototype1'])
    #         proto2 = self.extend_prefix(row['prototype2'])
    #         edge_type = self.extend_prefix(row['role'])
    #         just = row['just']
    #
    #         super_edge_info = SUPER_EDGE_TEMPLATE.format(
    #             proto1=proto1, proto2=proto2, edge_type=edge_type, just=just)
    #         self.write(super_edge_info)
    #
    #     for idx, row in self.df_event_role.iterrows():
    #         proto1 = self.extend_prefix(row['prototype1'])
    #         proto2 = self.extend_prefix(row['prototype2'])
    #         edge_type = self.extend_prefix(row['role'])
    #         just = row['just']
    #
    #         super_edge_info = SUPER_EDGE_TEMPLATE.format(
    #             proto1=proto1, proto2=proto2, edge_type=edge_type, just=just)
    #         self.write(super_edge_info)


def process():

    logger.info('exporting entity clusters')

    temp_dir = os.path.join(config['temp_dir'], config['run_name'])
    output_dir = os.path.join(config['output_dir'], config['run_name'])
    os.makedirs(output_dir, exist_ok=True)

    infile = os.path.join(temp_dir, 'entity_cluster.h5')
    event_file = infile[:-len('entity_cluster.h5')] + 'event_cluster.h5'
    event_role_file = infile[:-len('entity_cluster.h5')] + 'event_role.h5'
    relation_file = infile[:-len('entity_cluster.h5')] + 'relation_cluster.h5'
    relation_role_file = infile[:-len('entity_cluster.h5')] + 'relation_role.h5'
    outfile = os.path.join(output_dir, 'ta2_entity_cluster.ttl')
    exporter = Exporter(infile, event_file, event_role_file, relation_file, relation_role_file, outfile)
    exporter.run()

    # # assign bnode globally unique id
    # logger.info('assigning unique id for bnodes')
    # counter = [0]
    # re_bnode = re.compile(r'_:([A-Za-z0-9]+)')
    # ta1_concatenated_file = os.path.join(config['temp_dir'], config['run_name'], 'ta1_concatenated.nt')
    # for infile in glob.glob(os.path.join(config['temp_dir'], config['run_name'], '*/*.nt')):
    #     source = os.path.basename(infile).split('.')[0]
    #     print(infile)
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

    # exec_sh('apache-jena-3.16.0/bin/riot --syntax=ttl --output=nt < {ttl} > {nt}'
    #         .format(ttl=outfile, nt=outfile + '.nt'), logger)

    # # merge with ta1 output
    logger.info('merging ta1 outputs')
    # output_file = os.path.join(output_dir, 'ta2_named.ttl')

    # exec_sh('cat {temp_dir}/*/*.cleaned.nt > {output_dir}/ta1.ttl'
    #         .format(temp_dir=temp_dir, output_dir=output_dir), logger)
    # exec_sh('cat {output_dir}/ta1.ttl {output_dir}/ta2_entity_cluster.ttl > {output_dir}/ta2_named.ttl'
    #         .format(temp_dir=temp_dir, output_dir=output_dir), logger)
    # # not sure why file got truncated, no matter it is through os.system, sh.sh or subprocess.Popen
    # exec_sh('echo "cat {output_dir}/ta1.ttl {output_dir}/ta2_entity_cluster.ttl > {output_dir}/ta2_named.ttl" > {output_dir}/merge.sh'
    #       .format(temp_dir=temp_dir, output_dir=output_dir), logger)
    # exec_sh('chmod u+x {output_dir}/merge.sh; {output_dir}/merge.sh; rm {output_dir}/merge.sh'.format(output_dir=output_dir), logger)
    # # remove temp files
    # # exec_sh('rm {output_dir}/ta1.ttl {output_dir}/ta2_entity_cluster.ttl'
    # #         .format(output_dir=output_dir), logger)

    sh_cmd = f'''
    # merge ta1 files
    cat {temp_dir}/*/*.cleaned.nt > {output_dir}/ta1.ttl

    # merge ta1 with ta2
    cat {output_dir}/ta1.ttl {output_dir}/ta2_entity_cluster.ttl > {output_dir}/ta2_named.ttl

    # remove temp files
    rm {output_dir}/ta1.ttl {output_dir}/ta2_entity_cluster.ttl
    '''
    exec_sh(sh_cmd, logger)

    # sh_cmd = f'''
    # # merge ta1 files
    # # cat find . -regex '.*/{temp_dir}/[0-9a-zA-Z]*\.nt'
    # cat {temp_dir}/L0C04C6AJ/L0C04C6AJ.nt > {output_dir}/ta1_nt.ttl
    # '''
    # print(exec_sh(sh_cmd, logger))

    # with open(output_file, 'w') as fout:
    #     # ta1
    #     for ta1_file in glob.glob(f'{temp_dir}/*/*.cleaned.nt'):
    #         with open(ta1_file) as fin:
    #             for line in fin:
    #                 fout.write(line)
    #
    #     fout.write('\n\n')
    #
    #     # ta2
    #     with open(outfile) as fin:
    #         fout.write(fin.read())



if __name__ == '__main__':
    argv = sys.argv
    if argv[1] == 'process':
        process()
