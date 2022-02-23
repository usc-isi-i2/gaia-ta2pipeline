import os
import shutil
import json
import gzip
import csv
import sys
from collections import defaultdict
import glob
import warnings
import pandas as pd
import pyrallel
from config import config, get_logger
from common import exec_sh
import re
import rdflib


# ldc_kg = None
# df_wd_fb = None
# kb_to_fb_mapping = None
kgtk_labels = {}

re_cluster = re.compile(r'<.*InterchangeOntology#(clusterMember|ClusterMembership|SameAsCluster|cluster|prototype)>')
re_entity = re.compile(r'<.*InterchangeOntology#(Event|Entity|Relation)>')
re_bnode = re.compile(r'_:([^\s]*)')


class Importer(object):

    def __init__(self, source):
        self.source = source
        self.logger = get_logger('importer-' + source)
        self.infile = os.path.join(config['input_dir'], config['run_name'], '{}.ttl'.format(source))
        self.temp_dir = os.path.join(config['temp_dir'], config['run_name'], source)
        self.stat_info = {}

    def run(self):
        # global ldc_kg, df_wd_fb, kb_to_fb_mapping
        os.makedirs(self.temp_dir, exist_ok=True)

        try:

            nt_file = os.path.join(self.temp_dir, '{}.nt'.format(self.source))
            cleaned_nt_file = os.path.join(self.temp_dir, '{}.cleaned.nt'.format(self.source))
            kgtk_file = os.path.join(self.temp_dir, '{}.tsv'.format(self.source))
            kgtk_db_file = os.path.join(self.temp_dir, '{}.sqlite'.format(self.source))
            unreified_kgtk_file = kgtk_file + '.unreified'
            entity_outfile = os.path.join(self.temp_dir, '{}.entity.h5'.format(self.source))
            event_outfile = os.path.join(self.temp_dir, '{}.event.h5'.format(self.source))
            event_role_outfile = os.path.join(self.temp_dir, '{}.event_role.h5'.format(self.source))
            relation_outfile = os.path.join(self.temp_dir, '{}.relation.h5'.format(self.source))
            relation_role_outfile = os.path.join(self.temp_dir, '{}.relation_role.h5'.format(self.source))

            self.convert_ttl_to_nt(self.infile, nt_file)
            self.clean_nt(nt_file, cleaned_nt_file)
            self.convert_nt_to_kgtk(nt_file, kgtk_file)
            # self.unreify_kgtk(kgtk_file, unreified_kgtk_file)
            self.create_entity_df(kgtk_file, kgtk_db_file, entity_outfile, self.source)
            # self.create_event_df(kgtk_file, unreified_kgtk_file, event_outfile, self.source)
            # self.create_event_role_df(kgtk_file, unreified_kgtk_file, event_role_outfile, self.source,
            #                           entity_outfile, event_outfile)
            # self.create_relation_df(kgtk_file, unreified_kgtk_file, relation_outfile, self.source)
            # self.create_relation_role_df(kgtk_file, unreified_kgtk_file, relation_role_outfile, self.source,
            #                              entity_outfile, event_outfile, relation_outfile)

        except:
            self.logger.exception('Exception caught in Importer.run()')

        # os.remove(kgtk_file)
        # os.remove(unreified_kgtk_file)
        self.clean_temp_files()
        os.remove(kgtk_db_file)

    def create_namespace_file(self, outfile):
        os.makedirs(self.temp_dir, exist_ok=True)
        nt_file = os.path.join(self.temp_dir, '{}.nt'.format(self.source))
        kgtk_file = os.path.join(self.temp_dir, '{}.tsv'.format(self.source))
        self.convert_ttl_to_nt(self.infile, nt_file)
        exec_sh('''kgtk import-ntriples -i {nt_file} > {kgtk_file}'''
                     .format(nt_file=nt_file, kgtk_file=kgtk_file), self.logger)
        shutil.copy(kgtk_file, outfile)

    def tmp_file_path(self, x=None):
        suffix = '' if not x else '.{}'.format(x)
        return os.path.join(self.temp_dir, 'tmp{}'.format(suffix))

    def clean_temp_files(self):
        for f in glob.glob(os.path.join(self.temp_dir, 'tmp*')):
            os.remove(f)

    def predicate_path(self, dbfile, infile, path, quoting=0, doublequote=True):
        all_p = path.split('/')
        all_p = [f'-[:`{p}`]->' for p in all_p]
        all_p_str = ''.join([f'{all_p[idx]}(t{idx})' for idx in range(len(all_p)-1)]) \
                    + all_p[-1]  # create temp nodes in the middle

        exec_sh('kgtk query --graph-cache "{dbfile}" -i "{infile}" --match \'(s){p}(o)\' --return \'s,o\' > {tmp_file}'
                .format(dbfile=dbfile, infile=infile, p=all_p_str, tmp_file=self.tmp_file_path()), self.logger)
        pd_tmp = pd.read_csv(self.tmp_file_path(), delimiter='\t', quoting=quoting, doublequote=doublequote)
        return pd_tmp

    def kgtk_query(self, dbfile, infile, match, return_=None, where=None):
        query = f'kgtk query --graph-cache "{dbfile}" -i "{infile}"'

        if match:
            query += f' --match \'{match}\''
        if where:
            query += f' --where \'{where}\''
        if return_:
            query += f' --return \'{return_}\''

        query += f' > {self.tmp_file_path()}'
        # print(query)
        exec_sh(query, self.logger)
        pd_tmp = pd.read_csv(self.tmp_file_path(), delimiter='\t')
        return pd_tmp

    def convert_ttl_to_nt(self, ttl_file, nt_file):
        self.logger.info('converting ttl to nt')
        exec_sh('apache-jena-3.16.0/bin/riot --syntax=ttl --output=nt < {ttl} > {nt}'
                     .format(ttl=ttl_file, nt=self.tmp_file_path()), self.logger)

        # normalization (make iri globally unique)
        with open(self.tmp_file_path(), 'r') as fin:
            with open(nt_file, 'w') as fout:
                for line in fin:
                    line = line.strip()
                    # normalize bnode
                    line = re_bnode.sub(f'<http://www.isi.edu/gaia/bnode/{self.source}/' + r'\1' + '>', line)

                    # make cmu id globally unique
                    if config['enable_cmu_gid_patch']:
                        line = line.replace(
                            'http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/',
                            'http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/{}-'.format(self.source)
                        )

                    fout.write(line + '\n')

    def clean_nt(self, nt_file, cleaned_nt_file):
        # remove conflict TA1 triples
        self.logger.info('cleaning nt')

        # remove clusters
        self.logger.info('Loading TA1 graph')
        tmp_graph = rdflib.Graph()

        # load ns
        with open(config['namespace_file'], 'r') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                tmp_graph.bind(row['node1'], row['node2'])
        # for n in tmp_graph.namespace_manager.namespaces():
        #     print(n)

        tmp_graph.parse(nt_file, format='ttl')

        # remove associatedKEs
        self.logger.info('Removing TA1 associatedKEs')
        tmp_graph.update('''
        DELETE {
            ?claim aida:associatedKEs ?cluster .
        }
        WHERE {
            ?cluster a aida:SameAsCluster .
            ?cluster aida:prototype ?proto.
            ?proto a aida:Entity .
            
            ?claim a aida:Claim .
            ?claim aida:associatedKEs ?cluster .
        }''')

        # remove claimSemantics
        self.logger.info('Removing TA1 claimSemantics')
        tmp_graph.update('''
        DELETE {
            ?claim aida:claimSemantics ?cluster .
        }
        WHERE {
            ?cluster a aida:SameAsCluster .
            ?cluster aida:prototype ?proto.
            ?proto a aida:Entity .

            ?claim a aida:Claim .
            ?claim aida:claimSemantics ?cluster .
        }''')

        # remove cluster member
        self.logger.info('Removing TA1 ClusterMembership')
        tmp_graph.update('''
        DELETE {
            ?cm a aida:ClusterMembership .
            ?cm aida:cluster ?cluster .
        }
        WHERE {
            ?cluster a aida:SameAsCluster .
            ?cluster aida:prototype ?proto.
            ?proto a aida:Entity .

            ?cm a aida:ClusterMembership .
            ?cm aida:cluster ?cluster .
        }''')

        # remove cluster & prototype
        self.logger.info('Removing TA1 SameAsCluster')
        tmp_graph.update('''
        DELETE {
            ?cluster a aida:SameAsCluster .
            ?cluster aida:prototype ?proto.
        }
        WHERE {
            ?cluster a aida:SameAsCluster .
            ?cluster aida:prototype ?proto.
            ?proto a aida:Entity .
        }''')

        # result = tmp_graph.query('''
        # SELECT ?entity WHERE {
        #     ?entity a aida:Entity .
        # }''')
        # for row in result:
        #     print(row)

        tmp_graph.serialize(cleaned_nt_file, format='nt')

    def convert_nt_to_kgtk(self, nt_file, kgtk_file):
        self.logger.info('convert nt to kgtk')
        exec_sh('''kgtk import-ntriples \
      --namespace-file {ns_file} \
      --namespace-id-use-uuid False \
      --newnode-use-uuid False \
      --build-new-namespaces=False \
      --local-namespace-use-uuid True \
      --local-namespace-prefix {prefix} \
      --local-namespace-use-uuid False \
      -i {nt_file} > {kgtk_file}'''
        .format(ns_file=config['namespace_file'], prefix=self.source,  # prefix here would produce an invalid triple files
                nt_file=nt_file, kgtk_file=kgtk_file), self.logger)

    def unreify_kgtk(self, infile, outfile):
        self.logger.info('unreify kgtk')
        exec_sh('kgtk unreify-rdf-statements -i {infile} / sort --columns 1,2 >  {outfile}'
                .format(infile=infile, outfile=outfile), self.logger)

    def merge_values(self, values):
        # print(type(values))
        # print(values, values.index > 0)
        ret = {}
        for col in values.columns:
            if not values.empty:
                ret[col] = tuple(values[col].tolist())
            else:
                ret[col] = tuple([])
        return pd.Series(ret)

    def assign_qnode_label(self, value):
        global kgtk_labels
        return tuple([kgtk_labels.get(v) for v in value])

    def create_entity_df(self, kgtk_file, kgtk_db_file, output_file, source):
        self.logger.info('create entity df for ' + source)

        ### id
        self.logger.info('creating id')
        df_entity = self.kgtk_query(kgtk_db_file, kgtk_file,
            match='(e)-[:`rdf:type`]->(:`aida:Entity`)',
            return_='e AS e'
        )
        df_entity = df_entity.drop_duplicates().reset_index(drop=True)

        ### type
        self.logger.info('creating type')

        df_type = self.kgtk_query(kgtk_db_file, kgtk_file,
            match='(stmt)-[:`rdf:type`]->(:`rdf:Statement`),'+
                  '(stmt)-[:`rdf:subject`]->(e),'+
                  '(stmt)-[:`rdf:predicate`]->(:`rdf:type`),'+
                  '(stmt)-[:`rdf:object`]->(type),'+
                  '(stmt)-[:`aida:confidence`]->(c)-[:`aida:confidenceValue`]->(cv)',
            return_='e AS e,type AS type ,cv AS type_cv'
        )
        df_type = pd.merge(df_entity, df_type, left_on='e', right_on='e')
        df_type = df_type.groupby('e')[['type', 'type_cv']].apply(self.merge_values).reset_index()

        ### assign type label
        self.logger.info('assigning type label')
        df_type['type_label'] = df_type['type'].apply(self.assign_qnode_label)

        ### confidence
        self.logger.info('creating confidence')
        df_confidence = self.predicate_path(kgtk_db_file, kgtk_file, 'aida:confidence/aida:confidenceValue')\
            .rename(columns={'node1': 'e', 'node2': 'cv'})
        df_confidence = pd.merge(df_entity, df_confidence, left_on='e', right_on='e')

        ### name
        self.logger.info('creating name')
        df_name = self.predicate_path(kgtk_db_file, kgtk_file, 'aida:hasName')\
            .rename(columns={'node1': 'e', 'node2': 'name'})
        df_name = pd.merge(df_entity, df_name, left_on='e', right_on='e')
        df_name = df_name.groupby('e')[['name']].apply(self.merge_values).reset_index()

        ### link
        self.logger.info('creating link')
        df_link = self.kgtk_query(kgtk_db_file, kgtk_file,
            match='(e)-[:`aida:link`]->(t1)-[:`aida:linkTarget`]->(link),'+
                  '(e)-[:`aida:link`]->(t1)-[:`aida:confidence`]->(t2)-[:`aida:confidenceValue`]->(cv)',
            return_='e AS e,link AS link,cv AS link_cv'
        )
        df_link = pd.merge(df_entity, df_link, left_on='e', right_on='e')
        df_link = df_link.groupby('e')[['link', 'link_cv']].apply(self.merge_values).reset_index()

        ### assign link label
        self.logger.info('assigning type label')
        df_link['link_label'] = df_link['link'].apply(self.assign_qnode_label)

        ### informative justification
        self.logger.info('creating informative justification')
        df_infojust = self.predicate_path(kgtk_db_file, kgtk_file, 'aida:informativeJustification') \
            .rename(columns={'node1': 'e', 'node2': 'info_just'})
        df_infojust = pd.merge(df_entity, df_infojust, left_on='e', right_on='e')

        ### associated claims
        self.logger.info('creating associated claims')
        df_asso_claim = self.kgtk_query(kgtk_db_file, kgtk_file,
                                  match='(cluster)-[:`rdf:type`]->(:`aida:SameAsCluster`),'+
                                        '(cluster)-[:`aida:prototype`]->(proto)-[:`rdf:type`]->(:`aida:Entity`),'+
                                        '(cm)-[:`rdf:type`]->(:`aida:ClusterMembership`),'+
                                        '(cm)-[:`aida:cluster`]->(cluster),'+
                                        '(cm)-[:`aida:clusterMember`]->(e),'+
                                        '(claim)-[:`rdf:type`]->(:`aida:Claim`),'+
                                        '(claim)-[:`aida:associatedKEs`]->(cluster)',
                                  return_='e AS e, claim AS asso_claim'
                                  )
        df_asso_claim = pd.merge(df_entity, df_asso_claim, left_on='e', right_on='e')
        df_asso_claim = df_asso_claim.groupby('e')[['asso_claim']].apply(self.merge_values).reset_index()

        ### claim semantics
        self.logger.info('creating claim semantics')
        df_claim_seman = self.kgtk_query(kgtk_db_file, kgtk_file,
                                        match='(cluster)-[:`rdf:type`]->(:`aida:SameAsCluster`),'+
                                              '(cluster)-[:`aida:prototype`]->(proto)-[:`rdf:type`]->(:`aida:Entity`),'+
                                              '(cm)-[:`rdf:type`]->(:`aida:ClusterMembership`),'+
                                              '(cm)-[:`aida:cluster`]->(cluster),'+
                                              '(cm)-[:`aida:clusterMember`]->(e),'+
                                              '(claim)-[:`rdf:type`]->(:`aida:Claim`),'+
                                              '(claim)-[:`aida:claimSemantics`]->(cluster)',
                                        return_='e AS e, claim AS claim_seman'
                                        )
        df_claim_seman = pd.merge(df_entity, df_claim_seman, left_on='e', right_on='e')
        df_claim_seman = df_claim_seman.groupby('e')[['claim_seman']].apply(self.merge_values).reset_index()

        ### merge
        self.logger.info('merging all dfs to entity df')
        df_entity_complete = df_entity
        df_entity_complete = pd.merge(df_entity_complete, df_type, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_confidence, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_name, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_link, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_infojust, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_asso_claim, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_claim_seman, how='left')
        df_entity_complete['source'] = source
        df_entity_complete.drop_duplicates(subset=['e']).reset_index(drop=True)

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_entity_complete.to_hdf(output_file, 'entity', mode='w', format='fixed')
            df_entity_complete.to_csv(output_file + '.csv')

    def create_event_df(self, kgtk_file, unreified_kgtk_file, output_file, source):
        self.logger.info('creating event df for ' + source)

        ### id
        self.logger.info('creating id')
        exec_sh('kgtk filter -p ";rdf:type;aida:Event" -i {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_event = pd.read_csv(self.tmp_file_path(), delimiter='\t').drop(columns=['node2', 'label'])\
            .rename(columns={'node1': 'e'})
        # if self.stat_info['event'] != len(df_event):
        #     self.logger.error('TA1 has {} events, TA2 has {} events'.format(self.stat_info['event'], len(df_event)))
        df_event = df_event.drop_duplicates().reset_index(drop=True)

        ### type
        self.logger.info('creating type')
        exec_sh('kgtk filter -p ";rdf:type;" -i {kgtk_file} | kgtk filter --invert -p ";;aida:Event" > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_tmp1 = pd.read_csv(self.tmp_file_path(), delimiter='\t').rename(columns={'node1': 'e', 'node2': 'type'})
        df_event_type = pd.merge(df_event, df_tmp1, left_on='e', right_on='e').drop(columns=['label', 'id'])

        ### merge
        df_event_complete = pd.merge(df_event, df_event_type, how='left')
        df_event_complete['source'] = source
        df_event_complete.drop_duplicates(subset=['e']).reset_index(drop=True)

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_event_complete.to_hdf(output_file, 'event', mode='w', format='fixed')
            df_event_complete.to_csv(output_file + '.csv')

    def create_event_role_df(self, kgtk_file, unreified_kgtk_file, output_file, source, entity_file, event_file):
        self.logger.info('creating event role df for ' + source)
        exec_sh('kgtk filter --invert -p ";rdf:type;" -i {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        exec_sh("awk -F'\t' '$2 ~ /^ldcOnt:/' {tmp_file} > {tmp_file1}"
                     .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)), self.logger)
        df_event_role = pd.DataFrame(columns=['event', 'role', 'entity'])

        try:
            # entity ids and relation ids
            df_entity = pd.read_hdf(entity_file)['e']
            entity_ids = set([v for v in df_entity.to_dict().values()])
            df_event = pd.read_hdf(event_file)['e']
            event_ids = set([v for v in df_event.to_dict().values()])

            df_event_role = pd.read_csv(self.tmp_file_path(1),
                delimiter='\t', index_col=False, header=None, names=['event', 'role', 'entity', 'statement'])
            df_event_role['source'] = source

            df_event_role = df_event_role.loc[df_event_role['entity'].isin(entity_ids)]
            df_event_role = df_event_role.loc[df_event_role['event'].isin(event_ids)]
            df_event_role = df_event_role.drop_duplicates().reset_index(drop=True)

            # justified by
            exec_sh('kgtk filter -p ";aida:justifiedBy;" -i {kgtk_file} > {tmp_file}'
                    .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
            df_just = pd.read_csv(self.tmp_file_path(), delimiter='\t')
            just_dict = {v['node1']: v['node2'] for _, v in df_just.iterrows()}
            df_event_role['just'] = None
            df_event_role['just'] = df_event_role['statement'].apply(
                lambda x: '_:{}'.format(just_dict[x].split(':')[1]))
        except pd.errors.EmptyDataError:
            pass

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_event_role.to_hdf(output_file, 'event_role', mode='w', format='fixed')
            df_event_role.to_csv(output_file + '.csv')

    def create_relation_df(self, kgtk_file, unreified_kgtk_file, output_file, source):
        self.logger.info('creating relation df for ' + source)

        ### id
        self.logger.info('creating id')
        exec_sh('kgtk filter -p ";rdf:type;aida:Relation" -i {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_relation = pd.read_csv(self.tmp_file_path(), delimiter='\t').drop(columns=['node2', 'label']).rename(
            columns={'node1': 'e'})
        if self.stat_info['relation'] != len(df_relation):
            self.logger.error('TA1 has {} relations, TA2 has {} relations'.format(self.stat_info['relation'], len(df_relation)))
        df_relation = df_relation.drop_duplicates().reset_index(drop=True)

        ### type
        self.logger.info('creating type')
        exec_sh('kgtk filter -p ";rdf:type;" -i {kgtk_file} | kgtk filter --invert -p ";;aida:Relation" > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_tmp1 = pd.read_csv(self.tmp_file_path(), delimiter='\t').rename(columns={'node1': 'e', 'node2': 'type'})
        df_relation_type = pd.merge(df_relation, df_tmp1, left_on='e', right_on='e').drop(columns=['label', 'id'])

        ### merge
        df_relation_complete = pd.merge(df_relation, df_relation_type, how='left')
        df_relation_complete['source'] = source
        df_relation_complete = df_relation_complete.drop_duplicates(subset=['e']).reset_index(drop=True)

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_relation_complete.to_hdf(output_file, 'relation', mode='w', format='fixed')
            df_relation_complete.to_csv(output_file + '.csv')

    def create_relation_role_df(self, kgtk_file, unreified_kgtk_file, output_file, source,
                                entity_file, event_file, relation_file):
        self.logger.info('creating relation role df for ' + source)

        # role
        exec_sh('kgtk filter --invert -p ";rdf:type;" -i {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        exec_sh("awk -F'\t' '$2 ~ /^ldcOnt:/' {tmp_file} > {tmp_file1}"
                     .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)), self.logger)
        df_relation_role = pd.DataFrame(columns=['relation', 'role', 'entity'])

        try:
            # entity, event and relation ids
            df_entity = pd.read_hdf(entity_file)['e']
            entity_ids = set([v for v in df_entity.to_dict().values()])
            df_event = pd.read_hdf(event_file)['e']
            event_ids = set([v for v in df_event.to_dict().values()])
            df_relation = pd.read_hdf(relation_file)['e']
            relation_ids = set([v for v in df_relation.to_dict().values()])

            # read relations
            df_relation_role = pd.read_csv(self.tmp_file_path(1),
                delimiter='\t', index_col=False, header=None, names=['relation', 'role', 'e', 'statement'])
            df_relation_role['source'] = source

            df_relation_role = df_relation_role.loc[df_relation_role['relation'].isin(relation_ids)]
            df_relation_role_entity = df_relation_role.loc[df_relation_role['e'].isin(entity_ids)]
            df_relation_role_entity['type'] = 'entity'
            df_relation_role_event = df_relation_role.loc[df_relation_role['e'].isin(event_ids)]
            df_relation_role_event['type'] = 'event'
            df_relation_role = pd.concat([df_relation_role_entity, df_relation_role_event], ignore_index=True)
            df_relation_role = df_relation_role.drop_duplicates().reset_index(drop=True)

            # justified by
            exec_sh('kgtk filter -p ";aida:justifiedBy;" -i {kgtk_file} > {tmp_file}'
                    .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
            df_just = pd.read_csv(self.tmp_file_path(), delimiter='\t')
            just_dict = {v['node1']: v['node2'] for _, v in df_just.iterrows()}
            df_relation_role['just'] = None
            df_relation_role['just'] = df_relation_role['statement'].apply(
                lambda x: '_:{}'.format(just_dict[x].split(':')[1]))
        except pd.errors.EmptyDataError:
            pass

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_relation_role.to_hdf(output_file, 'relation_role', mode='w', format='fixed')
            df_relation_role.to_csv(output_file + '.csv')


def load_resource():
    global kgtk_labels
    with gzip.open(config['kgtk_labels'], 'rt') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            kgtk_labels[row['node1']] = row['node2']


# def convert_nan_to_none(df):
#     return df.where(pd.notnull(df), None)
#
#
# def create_wd_to_fb_mapping():
#     # only need to run once
#     # step 1
#     # kgtk import-wikidata \
#     #      -i wikidata-20200803-all.json.bz2 \
#     #      --node wikidata-20200803-all-nodes.tsv \
#     #      --edge wikidata-20200803-all-edges.tsv \
#     #      --qual wikidata-20200803-all-qualifiers.tsv \
#     #      --explode-values False \
#     #      --lang en,ru,uk \
#     #      --procs 24 \
#     #      |& tee wikidata-20200803-all-import.log
#     # step 2
#     # kgtk filter -p ';P646;' wikidata-20200803-all-edges.tsv.gz > qnode_to_freebase_20200803.tsv
#     # step 3
#     # kgtk ifexists --filter-on qnode_to_freebase.tsv --filter-keys node1 \
#     # --input-keys id -i wikidata-20200803-all-nodes.tsv.gz > qnode_freebase_20200803.tsv
#
#     # df_wdid_fb = pd.read_csv('qnode_to_free_tsv_file_path', delimiter='\t').drop(columns=['id', 'label', 'rank'])\
#     #     .rename(columns={'node1': 'qnode', 'node2': 'fbid'})
#     # df_wd_node = pd.read_csv('filtered_qnode_file_path', delimiter='\t').drop(columns={'type'})\
#     #     .rename(columns={'id': 'qnode'})
#     # df_wd_fb = pd.merge(df_wd_node, df_wdid_fb, left_on='qnode', right_on='qnode')
#     # df_wd_fb.to_csv(config['wd_to_fb_file'], index=False)
#     pass
#
#
# def load_ldc_kb():
#     kb_names = defaultdict(lambda: {'type': None, 'names': []})
#
#     # entities
#     with open(os.path.join(config['ldc_kg_dir'], 'entities.tab')) as f:
#         for idx, line in enumerate(f):
#             if idx == 0:
#                 continue
#             line = line.strip().split('\t')
#             type_, id_, name1 = line[1], line[2], line[3]
#             kb_names[id_]['type'] = type_
#             kb_names[id_]['names'].append(name1)
#             if len(line) >= 5:
#                 name2 = line[4]
#                 kb_names[id_]['names'].append(name2)
#
#     # alternative names
#     with open(os.path.join(config['ldc_kg_dir'], 'alternate_names.tab')) as f:
#         for idx, line in enumerate(f):
#             if idx == 0:
#                 continue
#             line = line.strip().split('\t')
#             id_, name_ = line[0], line[1]
#             kb_names[id_]['names'].append(name_)
#
#     return kb_names
#
#
# def load_kb_to_fb_mapping():
#     mapping = None
#     if config['kb_to_fbid_mapping']:
#         with open(config['kb_to_fbid_mapping'], 'r') as f:
#             mapping = json.load(f)
#     return mapping
#
#
# def load_wd_to_fb_df():
#     return convert_nan_to_none(pd.read_csv(config['wd_to_fb_file']))


def worker(source):
    importer = Importer(source=source)
    importer.run()


def process():
    # global ldc_kg, df_wd_fb, kb_to_fb_mapping
    logger = get_logger('importer-main')
    logger.info('loading resource')
    load_resource()
    # ldc_kg = load_ldc_kb()
    # df_wd_fb = load_wd_to_fb_df()
    # kb_to_fb_mapping = load_kb_to_fb_mapping()

    logger.info('starting multiprocessing mode')
    pp = pyrallel.ParallelProcessor(
        num_of_processor=config['num_of_processor'],
        mapper=worker,
        max_size_per_mapper_queue=config['num_of_processor'] * 2
    )
    pp.start()

    all_infiles = glob.glob(os.path.join(config['input_dir'], config['run_name'], '*.ttl'))
    logger.info(f'{len(all_infiles)} files to process')
    for idx, infile in enumerate(all_infiles):
        source = os.path.basename(infile).split('.')[0]
        pp.add_task(source)
        logger.info(f'adding task {source} [{idx+1}/{len(all_infiles)}]')

    pp.task_done()
    pp.join()
    logger.info('all tasks are finished')

    # integrity check
    # logger.info('checking file integrity')
    # all_ta1_files = set()
    # all_ta2_nt_files = set()
    # for infile in glob.glob(os.path.join(config['input_dir'], config['run_name'], '*.ttl')):
    #     source = os.path.basename(infile).split('.')[0]
    #     all_ta1_files.add(source)
    # for infile in glob.glob(os.path.join(config['temp_dir'], config['run_name'], '*/*.cleaned.nt')):
    #     source = os.path.basename(infile).split('.')[0]
    #     all_ta2_nt_files.add(source)
    #     fn = os.path.join(config['temp_dir'], config['run_name'], source, source)
    #     if not os.path.exists(fn + '.tsv'):
    #         logger.error('Incorrect KGTK file: {}'.format(source))
        # if not os.path.exists(fn + '.entity.h5'):
        #     logger.error('Incorrect entity df: {}'.format(source))
        # if not os.path.exists(fn + '.event.h5'):
        #     logger.error('Incorrect event df: {}'.format(source))
        # if not os.path.exists(fn + '.relation.h5'):
        #     logger.error('Incorrect relation df: {}'.format(source))
        # if not os.path.exists(fn + '.event_role.h5'):
        #     logger.error('Incorrect event role df: {}'.format(source))
        # if not os.path.exists(fn + '.relation_role.h5'):
        #     logger.error('Incorrect relation role df: {}'.format(source))
    # ta2_missing = all_ta1_files - all_ta2_nt_files
    # if len(ta2_missing) > 0:
    #     for source in ta2_missing:
    #         logger.error('{} has not been parsed'.format(source))
    # logger.info('integrity check completed')


# def generate_kb_to_wd_mapping(run_name, outfile):
#     df_entity = pd.DataFrame()
#     for infile in glob.glob(os.path.join(config['temp_dir'], run_name, '*/*.entity.h5')):
#         df_entity = df_entity.append(pd.read_hdf(infile))
#     df_entity = df_entity.reset_index(drop=True)
#
#     mapping = defaultdict(lambda: defaultdict(float))
#     for idx, e in df_entity.iterrows():
#         targets = e['target']
#         target_scores = e['target_score']
#         fbs = e['fbid']
#         fb_scores = e['fbid_score_avg']
#         if pd.notna(targets) and pd.notna(fbs):
#             for i, t in enumerate(targets):
#                 t_score = target_scores[i]
#                 for j, fb in enumerate(fbs):
#                     fb_score = fb_scores[j]
#                     curr_score = 1.0 * t_score * fb_score
#                     prev_score = mapping[t].get(fb)
#                     if prev_score:
#                         mapping[t][fb] = max(curr_score, prev_score)
#                     else:
#                         mapping[t][fb] = curr_score
#     with open(outfile, 'w') as f:
#         json.dump(mapping, f)


if __name__ == '__main__':
    argv = sys.argv
    if argv[1] == 'process':
        process()
    # elif argv[1] == 'kb_to_wd':
    #     run_name = argv[2]
    #     outfile = argv[3]
    #     generate_kb_to_wd_mapping(outfile)
    # elif argv[1] == 'create_namespace':
    #     outfile = argv[2]
    #
    #     # pick the file with biggest size
    #     source = None
    #     source_size = 0
    #     for infile in glob.glob(os.path.join(config['input_dir'], config['run_name'], '*.ttl')):
    #         if not source:
    #             source = infile
    #         file_size = os.stat(infile).st_size
    #         if file_size > source_size:
    #             source = os.path.basename(infile).split('.')[0]
    #     im = Importer(source=source)
    #     im.create_namespace_file(outfile)
