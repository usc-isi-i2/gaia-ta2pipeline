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
        self.infile = os.path.join(config['input_dir'], config['run_name'], config['subrun_name'], f'{source}.ttl')
        self.temp_dir = os.path.join(config['temp_dir'], config['run_name'], config['subrun_name'], source)
        self.stat_info = {}

    def run(self):
        # global ldc_kg, df_wd_fb, kb_to_fb_mapping
        os.makedirs(self.temp_dir, exist_ok=True)

        try:

            nt_file = os.path.join(self.temp_dir, '{}.nt'.format(self.source))
            cleaned_nt_file = os.path.join(self.temp_dir, '{}.cleaned.nt'.format(self.source))
            kgtk_file = os.path.join(self.temp_dir, '{}.tsv'.format(self.source))
            kgtk_db_file = os.path.join(self.temp_dir, '{}.sqlite'.format(self.source))
            entity_outfile = os.path.join(self.temp_dir, '{}.entity.h5'.format(self.source))
            event_outfile = os.path.join(self.temp_dir, '{}.event.h5'.format(self.source))
            relation_outfile = os.path.join(self.temp_dir, '{}.relation.h5'.format(self.source))
            role_outfile = os.path.join(self.temp_dir, '{}.role.h5'.format(self.source))

            self.convert_ttl_to_nt(self.infile, nt_file)
            self.clean_nt(nt_file, cleaned_nt_file)
            self.convert_nt_to_kgtk(nt_file, kgtk_file)
            self.create_entity_df(kgtk_file, kgtk_db_file, entity_outfile, self.source)
            self.create_event_df(kgtk_file, kgtk_db_file, event_outfile, self.source)
            self.create_relation_df(kgtk_file, kgtk_db_file, relation_outfile, self.source)
            self.create_role(kgtk_file, kgtk_db_file, role_outfile, self.source)

        except:
            self.logger.exception('Exception caught in Importer.run()')

        os.remove(nt_file)
        os.remove(kgtk_file)
        os.remove(kgtk_db_file)
        self.clean_temp_files()

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

    def kgtk_query(self, dbfile, infile, match, option=None, return_=None, where=None, quoting=csv.QUOTE_MINIMAL):
        query = f'kgtk query --graph-cache "{dbfile}" -i "{infile}"'

        if match:
            query += f' --match \'{match}\''
        if where:
            query += f' --where \'{where}\''
        if option:
            for opt in option:
                query += f' --opt \'{opt}\''
        if return_:
            query += f' --return \'{return_}\''

        query += f' > {self.tmp_file_path()}'
        # print(query)
        exec_sh(query, self.logger)

        # kgtk query set quoting to csv.QUOTE_NONE by default
        # https://github.com/usc-isi-i2/kgtk/blob/6168e06fac121f2e60b687ff90ee6f5cc3d074b5/kgtk/cli/query.py#L288
        pd_tmp = pd.read_csv(self.tmp_file_path(), delimiter='\t', quoting=quoting)
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

                    fout.write(line + '\n')

    def execute_update(self, infile, query):
        query_file = self.tmp_file_path('query')
        tmp_outfile = self.tmp_file_path('out')
        with open(query_file, 'w') as f:
            f.write(query)
        exec_sh(f'apache-jena-3.16.0/bin/update --data={infile} --update={query_file} --dump > {tmp_outfile}', self.logger)
        shutil.move(tmp_outfile, infile)
        os.remove(query_file)

    def clean_nt(self, nt_file, cleaned_nt_file):
        # remove conflict TA1 triples
        self.logger.info('cleaning nt')

        # remove clusters
        self.logger.info('Loading TA1 graph')

        # load ns
        str_ns = ''
        with open(config['namespace_file'], 'r') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                str_ns += f'PREFIX {row["node1"]}: <{row["node2"]}>\n'

        # make a copy to work on
        shutil.copy(nt_file, cleaned_nt_file)

        # remove associatedKEs
        self.logger.info('Removing TA1 associatedKEs')
        str_update = '''
        DELETE {
            ?claim aida:associatedKEs ?cluster .
        }
        WHERE {
            ?cluster a aida:SameAsCluster .
            ?cluster aida:prototype ?proto.
            ?proto a aida:Entity .

            ?claim a aida:Claim .
            ?claim aida:associatedKEs ?cluster .
        }
        
        '''
        self.execute_update(cleaned_nt_file, str_ns + str_update)

        # remove claimSemantics
        self.logger.info('Removing TA1 claimSemantics')
        str_update = '''
        DELETE {
            ?claim aida:claimSemantics ?cluster .
        }
        WHERE {
            ?cluster a aida:SameAsCluster .
            ?cluster aida:prototype ?proto.
            ?proto a aida:Entity .

            ?claim a aida:Claim .
            ?claim aida:claimSemantics ?cluster .
        }'''
        self.execute_update(cleaned_nt_file, str_ns + str_update)

        # remove cluster member
        self.logger.info('Removing TA1 ClusterMembership')
        str_update = '''
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
        }'''
        self.execute_update(cleaned_nt_file, str_ns + str_update)

        # remove cluster & prototype
        self.logger.info('Removing TA1 SameAsCluster')
        str_update = '''
        DELETE {
            ?cluster a aida:SameAsCluster .
            ?cluster aida:prototype ?proto.
        }
        WHERE {
            ?cluster a aida:SameAsCluster .
            ?cluster aida:prototype ?proto.
            ?proto a aida:Entity .
        }'''
        self.execute_update(cleaned_nt_file, str_ns + str_update)

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
                  # '(e)-[:`aida:justifiedBy`]->(just)',
            return_='e AS e'
        )
        df_entity = df_entity.drop_duplicates().reset_index(drop=True)
        # df_entity = df_entity.groupby('e')[['e_just']].apply(self.merge_values).reset_index()

        ### type
        self.logger.info('creating type')

        df_type = self.kgtk_query(kgtk_db_file, kgtk_file,
            match='(stmt)-[:`rdf:type`]->(stmt_type),'+
                  '(stmt)-[:`rdf:subject`]->(e),'+
                  '(stmt)-[:`rdf:predicate`]->(:`rdf:type`),'+
                  '(stmt)-[:`rdf:object`]->(type),'+
                  '(stmt)-[:`aida:confidence`]->(c)-[:`aida:confidenceValue`]->(cv),'+
                  '(stmt)-[:`aida:justifiedBy`]->(just)',
            where='stmt_type IN ["rdf:Statement", "aida:TypeStatement"]',
            return_='e AS e,type AS type,cv AS type_cv,just AS type_just'
        )
        df_type = pd.merge(df_entity, df_type, left_on='e', right_on='e')
        df_type = df_type.groupby('e')[['type', 'type_cv', 'type_just']].apply(self.merge_values).reset_index()

        def merge_just(v):

            result = {'e': v['e'], 'type': [], 'type_cv': [], 'type_just': []}

            type_, type_cv, type_just = v['type'], v['type_cv'], v['type_just']
            unique_type = set(type_)
            for t in unique_type:
                # use the maximum cv
                # aggregate justification
                indices = [i for i, x in enumerate(type_) if x == t]
                cv = max([type_cv[i] for i in indices])
                justs = tuple([type_just[i] for i in indices])

                result['type'].append(t)
                result['type_cv'].append(cv)
                result['type_just'].append(justs)

            result['type'] = tuple(result['type'])
            result['type_cv'] = tuple(result['type_cv'])
            result['type_just'] = tuple(result['type_just'])
            return pd.Series(result)

        df_type = df_type.apply(merge_just, axis=1).reset_index(drop=True)

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
        df_infojust = self.kgtk_query(kgtk_db_file, kgtk_file,
                                    match='(e)-[:`rdf:type`]->(:`aida:Entity`),'+
                                          '(e)-[:`aida:informativeJustification`]->(ij)',
                                    return_='e AS e, ij AS info_just'
                                    )
        df_infojust = pd.merge(df_entity, df_infojust, left_on='e', right_on='e')

        ### informative justification extension
        if config.get('extract_mention', False):
            self.logger.info('creating informative justification extension')
            df_infojust_ext = self.kgtk_query(kgtk_db_file, kgtk_file,
                                          match='(e)-[:`rdf:type`]->(:`aida:Entity`),'+
                                                '(e)-[:`aida:informativeJustification`]->(ij),'+
                                                '(ij)-[:`rdf:type`]->(:`aida:TextJustification`),'+
                                                '(ij)-[:`aida:startOffset`]->(ij_start),'+
                                                '(ij)-[:`aida:endOffsetInclusive`]->(ij_end),'+
                                                '(ij)-[:`aida:privateData`]->(p),'+
                                                '(p)-[:`aida:jsonContent`]->(j),'+
                                                '(p)-[:`aida:system`]->(:`http://www.uiuc.edu/mention`)',
                                          return_='ij AS info_just, ij_start AS ij_start, ij_end AS ij_end, j AS mention',
                                          quoting=csv.QUOTE_NONE  # this maks mention string properly parsed
                                          )

            def parse_private_date(v):
                try:
                    v = json.loads(eval(v))
                    return v
                    # return v.get('mention_string')
                except:
                    return None

            df_infojust_ext['mention'] = df_infojust_ext['mention'].apply(parse_private_date)
            df_infojust = pd.merge(df_infojust, df_infojust_ext, left_on='info_just', right_on='info_just', how='left')


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

        ### cluster
        self.logger.info('creating associated cluster')
        df_cluster = self.kgtk_query(kgtk_db_file, kgtk_file,
                                     match='(cluster)-[:`rdf:type`]->(:`aida:SameAsCluster`),'+
                                           '(cluster)-[:`aida:prototype`]->(proto)-[:`rdf:type`]->(:`aida:Entity`),'+
                                           '(cm)-[:`rdf:type`]->(:`aida:ClusterMembership`),'+
                                           '(cm)-[:`aida:cluster`]->(cluster),'+
                                           '(cm)-[:`aida:clusterMember`]->(e)',
                                     return_='e AS e, proto AS ta1_proto, cluster AS ta1_cluster'
                                     )
        df_cluster = pd.merge(df_entity, df_cluster, left_on='e', right_on='e')
        df_cluster = df_cluster.groupby('e')[['ta1_proto', 'ta1_cluster']].apply(self.merge_values).reset_index()

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
        df_entity_complete = pd.merge(df_entity_complete, df_cluster, how='left')
        df_entity_complete['source'] = source
        df_entity_complete.drop_duplicates(subset=['e']).reset_index(drop=True)

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_entity_complete.to_hdf(output_file, 'entity', mode='w', format='fixed')
            df_entity_complete.to_csv(output_file + '.csv')

    def create_event_df(self, kgtk_file, kgtk_db_file, output_file, source):
        self.logger.info('create event df for ' + source)

        ### id
        self.logger.info('creating id')
        df_event = self.kgtk_query(kgtk_db_file, kgtk_file,
                                    match='(e)-[:`rdf:type`]->(:`aida:Event`)',
                                    return_='e AS e'
                                    )
        df_event = df_event.drop_duplicates().reset_index(drop=True)

        ### type
        self.logger.info('creating type')

        df_type = self.kgtk_query(kgtk_db_file, kgtk_file,
                                  match='(stmt)-[:`rdf:type`]->(stmt_type),'+
                                        '(stmt)-[:`rdf:subject`]->(e),'+
                                        '(stmt)-[:`rdf:predicate`]->(:`rdf:type`),'+
                                        '(stmt)-[:`rdf:object`]->(type),'+
                                        '(stmt)-[:`aida:confidence`]->(c)-[:`aida:confidenceValue`]->(cv)',
                                  where='stmt_type IN ["rdf:Statement", "aida:TypeStatement"]',
                                  return_='e AS e,type AS type,cv AS type_cv'
                                  )
        df_type = pd.merge(df_event, df_type, left_on='e', right_on='e')
        df_type = df_type.groupby('e')[['type', 'type_cv']].apply(self.merge_values).reset_index()
        df_type['type_label'] = df_type['type'].apply(self.assign_qnode_label)

        ### time
        self.logger.info('creating datetime')
        # df_time = self.kgtk_query(kgtk_db_file, kgtk_file,
        #                           match='(e)-[:`aida:ldcTime`]->(dt)',
        #                           option=('(dt)-[:`aida:end`]->(end)',
        #                                 '(end)-[:`aida:timeType`]->(dte_type)',
        #                                 '(end)-[:`aida:day`]->(e1)-[:`kgtk:structured_value`]->(dte_day)',
        #                                 '(end)-[:`aida:month`]->(e2)-[:`kgtk:structured_value`]->(dte_month)',
        #                                 '(end)-[:`aida:year`]->(e3)-[:`kgtk:structured_value`]->(dte_year)',
        #                                 '(dt)-[:`aida:start`]->(start)',
        #                                 '(start)-[:`aida:timeType`]->(dts_type)',
        #                                 '(start)-[:`aida:day`]->(s1)-[:`kgtk:structured_value`]->(dts_day)',
        #                                 '(start)-[:`aida:month`]->(s2)-[:`kgtk:structured_value`]->(dts_month)',
        #                                 '(start)-[:`aida:year`]->(s3)-[:`kgtk:structured_value`]->(dts_year)'),
        #                           return_='e AS e,'+
        #                                   'dte_type AS dte_type, dte_day AS dte_day, dte_month AS dte_month, dte_year AS dte_year,'+
        #                                   'dts_type AS dts_type, dts_day AS dts_day, dts_month AS dts_month, dts_year AS dts_year'
        #                           )


        def merge_time(values):
            output = []
            for idx, row in values.iterrows():
                output_inner = {}
                for k, v in row.items():
                    output_inner[k] = v
                output.append(output_inner)
            return pd.Series({'dt': output})

        df_time_end = self.kgtk_query(kgtk_db_file, kgtk_file,
                                    match='(e)-[:`aida:ldcTime`]->(dt)-[:`aida:end`]->(end)-[:`aida:timeType`]->(type)',  # dt_type: ON, BEFORE, AFTER, UNKNOWN
                                    option=('(end)-[:`aida:day`]->(e1)-[:`kgtk:structured_value`]->(day)',
                                          '(end)-[:`aida:month`]->(e2)-[:`kgtk:structured_value`]->(month)',
                                          '(end)-[:`aida:year`]->(e3)-[:`kgtk:structured_value`]->(year)'),
                                    return_='e AS e, type AS type, day AS day, month AS month, year AS year'
                                    )
        df_time_end = df_time_end.groupby('e')[['type', 'day', 'month', 'year']].apply(merge_time).rename(columns={'dt':'dt_end'}).reset_index()
        df_time_start = self.kgtk_query(kgtk_db_file, kgtk_file,
                                      match='(e)-[:`aida:ldcTime`]->(dt)-[:`aida:start`]->(start)-[:`aida:timeType`]->(type)',  # dt_type: ON, BEFORE, AFTER, UNKNOWN
                                      option=('(start)-[:`aida:day`]->(e1)-[:`kgtk:structured_value`]->(day)',
                                              '(start)-[:`aida:month`]->(e2)-[:`kgtk:structured_value`]->(month)',
                                              '(start)-[:`aida:year`]->(e3)-[:`kgtk:structured_value`]->(year)'),
                                      return_='e AS e, type AS type, day AS day, month AS month, year AS year'
                                      )
        df_time_start = df_time_start.groupby('e')[['type', 'day', 'month', 'year']].apply(merge_time).rename(columns={'dt':'dt_start'}).reset_index()

        df_time = pd.merge(df_time_start, df_time_end)

        # associated cluster
        self.logger.info('creating associated cluster')
        df_cluster = self.kgtk_query(kgtk_db_file, kgtk_file,
                                        match='(cluster)-[:`rdf:type`]->(:`aida:SameAsCluster`),'+
                                              '(cluster)-[:`aida:prototype`]->(proto)-[:`rdf:type`]->(:`aida:Event`),'+
                                              '(cm)-[:`rdf:type`]->(:`aida:ClusterMembership`),'+
                                              '(cm)-[:`aida:cluster`]->(cluster),'+
                                              '(cm)-[:`aida:clusterMember`]->(e)',
                                        return_='e AS e, proto AS proto, cluster AS cluster'
                                        )
        df_cluster = pd.merge(df_event, df_cluster, left_on='e', right_on='e')
        df_cluster = df_cluster.groupby('e')[['proto', 'cluster']].apply(self.merge_values).reset_index()

        ### merge
        self.logger.info('merging dfs')
        df_event_complete = df_event
        df_event_complete = pd.merge(df_event_complete, df_type, how='left')
        df_event_complete = pd.merge(df_event_complete, df_time, how='left')
        df_event_complete = pd.merge(df_event_complete, df_cluster, how='left')
        df_event_complete['source'] = source
        df_event_complete.drop_duplicates(subset=['e']).reset_index(drop=True)

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_event_complete.to_hdf(output_file, 'event', mode='w', format='fixed')
            df_event_complete.to_csv(output_file + '.csv')

    def create_relation_df(self, kgtk_file, kgtk_db_file, output_file, source):
        self.logger.info('create relation df for ' + source)

        ### id
        self.logger.info('creating id')
        df_relation = self.kgtk_query(kgtk_db_file, kgtk_file,
                                   match='(e)-[:`rdf:type`]->(:`aida:Relation`)',
                                   return_='e AS e'
                                   )
        df_relation = df_relation.drop_duplicates().reset_index(drop=True)

        ### type
        self.logger.info('creating type')

        df_type = self.kgtk_query(kgtk_db_file, kgtk_file,
                                  match='(stmt)-[:`rdf:type`]->(stmt_type),'+
                                        '(stmt)-[:`rdf:subject`]->(e),'+
                                        '(stmt)-[:`rdf:predicate`]->(:`rdf:type`),'+
                                        '(stmt)-[:`rdf:object`]->(type),'+
                                        '(stmt)-[:`aida:confidence`]->(c)-[:`aida:confidenceValue`]->(cv)',
                                  where='stmt_type IN ["rdf:Statement", "aida:TypeStatement"]',
                                  return_='e AS e,type AS type,cv AS type_cv'
                                  )
        df_type = pd.merge(df_relation, df_type, left_on='e', right_on='e')
        df_type = df_type.groupby('e')[['type', 'type_cv']].apply(self.merge_values).reset_index()

        # associated cluster
        self.logger.info('creating associated cluster')
        df_cluster = self.kgtk_query(kgtk_db_file, kgtk_file,
                                     match='(cluster)-[:`rdf:type`]->(:`aida:SameAsCluster`),'+
                                           '(cluster)-[:`aida:prototype`]->(proto)-[:`rdf:type`]->(:`aida:Relation`),'+
                                           '(cm)-[:`rdf:type`]->(:`aida:ClusterMembership`),'+
                                           '(cm)-[:`aida:cluster`]->(cluster),'+
                                           '(cm)-[:`aida:clusterMember`]->(e)',
                                     return_='e AS e, proto AS proto, cluster AS cluster'
                                     )
        df_cluster = pd.merge(df_relation, df_cluster, left_on='e', right_on='e')
        df_cluster = df_cluster.groupby('e')[['proto', 'cluster']].apply(self.merge_values).reset_index()

        ### merge
        self.logger.info('merging dfs')
        df_relation_complete = df_relation
        df_relation_complete = pd.merge(df_relation_complete, df_type, how='left')
        df_relation_complete = pd.merge(df_relation_complete, df_cluster, how='left')
        df_relation_complete['source'] = source
        df_relation_complete.drop_duplicates(subset=['e']).reset_index(drop=True)

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_relation_complete.to_hdf(output_file, 'relation', mode='w', format='fixed')
            df_relation_complete.to_csv(output_file + '.csv')

    def create_role(self, kgtk_file, kgtk_db_file, output_file, source):

        self.logger.info('creating role')

        # entities = set(pd.read_hdf(entity_outfile)['e'].to_list())
        # events = set(pd.read_hdf(event_outfile)['e'].to_list())
        # relations = set(pd.read_hdf(relation_outfile)['e'].to_list())

        df_role = self.kgtk_query(kgtk_db_file, kgtk_file,
                                  match='(stmt)-[:`rdf:type`]->(stmt_type),'+
                                        '(stmt)-[:`rdf:subject`]->(e1),'+
                                        '(stmt)-[:`rdf:predicate`]->(role),'+
                                        '(stmt)-[:`rdf:object`]->(e2),'+
                                        '(stmt)-[:`aida:confidence`]->(c)-[:`aida:confidenceValue`]->(cv),'+
                                        '(stmt)-[:`aida:justifiedBy`]->(just),'+
                                        '(e1)-[:`rdf:type`]->(e1_type),'+
                                        '(e2)-[:`rdf:type`]->(e2_type)',
                                  where='role != "rdf:type" AND stmt_type IN ["rdf:Statement", "aida:ArgumentStatement"]',
                                  return_='e1 AS e1, e2 AS e2, e1_type AS e1_type, e2_type AS e2_type, role AS role, cv AS cv, just AS just'
                                  )

        df_role['source'] = source

        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_role.to_hdf(output_file, 'role', mode='w', format='fixed')
            df_role.to_csv(output_file + '.csv')


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


def worker(source, logger=None, message=None):
    if logger and message:
        logger.info(message)
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

    all_infiles = glob.glob(os.path.join(config['input_dir'], config['run_name'], config['subrun_name'], '*.ttl'))
    logger.info(f'{len(all_infiles)} files to process')
    for idx, infile in enumerate(all_infiles):
        source = os.path.basename(infile).split('.')[0]
        pp.add_task(source, logger, f'starting task {source} [{idx+1}/{len(all_infiles)}]')

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
