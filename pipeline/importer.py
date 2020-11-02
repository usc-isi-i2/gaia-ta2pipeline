import os
import shutil
import json
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


ldc_kg = None
df_wd_fb = None
kb_to_fb_mapping = None

re_cluster = re.compile(r'<.*InterchangeOntology#(clusterMember|ClusterMembership|SameAsCluster|cluster|prototype)>')
re_entity = re.compile(r'<.*InterchangeOntology#(Event|Entity|Relation)>')


class Importer(object):

    def __init__(self, source):
        self.source = source
        self.logger = get_logger('importer-' + source)
        self.infile = os.path.join(config['input_dir'], config['run_name'], '{}.ttl'.format(source))
        self.temp_dir = os.path.join(config['temp_dir'], config['run_name'], source)
        self.stat_info = {}

    def run(self):
        global ldc_kg, df_wd_fb, kb_to_fb_mapping
        os.makedirs(self.temp_dir, exist_ok=True)

        try:

            nt_file = os.path.join(self.temp_dir, '{}.nt'.format(self.source))
            cleaned_nt_file = os.path.join(self.temp_dir, '{}.cleaned.nt'.format(self.source))
            kgtk_file = os.path.join(self.temp_dir, '{}.tsv'.format(self.source))
            unreified_kgtk_file = kgtk_file + '.unreified'
            entity_outfile = os.path.join(self.temp_dir, '{}.entity.h5'.format(self.source))
            event_outfile = os.path.join(self.temp_dir, '{}.event.h5'.format(self.source))
            event_role_outfile = os.path.join(self.temp_dir, '{}.event_role.h5'.format(self.source))
            relation_outfile = os.path.join(self.temp_dir, '{}.relation.h5'.format(self.source))
            relation_role_outfile = os.path.join(self.temp_dir, '{}.relation_role.h5'.format(self.source))

            self.convert_ttl_to_nt(self.infile, nt_file)
            self.clean_nt(nt_file, cleaned_nt_file)
            self.convert_nt_to_kgtk(cleaned_nt_file, kgtk_file)
            self.unreify_kgtk(kgtk_file, unreified_kgtk_file)
            self.create_entity_df(kgtk_file, unreified_kgtk_file, entity_outfile, self.source,
                                  ldc_kg, df_wd_fb, kb_to_fb_mapping)
            self.create_event_df(kgtk_file, unreified_kgtk_file, event_outfile, self.source)
            self.create_event_role_df(kgtk_file, unreified_kgtk_file, event_role_outfile, self.source,
                                      entity_outfile, event_outfile)
            self.create_relation_df(kgtk_file, unreified_kgtk_file, relation_outfile, self.source)
            self.create_relation_role_df(kgtk_file, unreified_kgtk_file, relation_role_outfile, self.source,
                                         entity_outfile, relation_outfile)

        except:
            self.logger.exception('Exception caught in Importer.run()')

        # os.remove(kgtk_file)
        # os.remove(unreified_kgtk_file)
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

    def predicate_path(self, infile, path, retain_intermediate=False):
        all_p = path.split('/')
        if len(all_p) == 0:
            return

        # first predicate
        tmp_str = ';{};'.format(all_p[0])
        exec_sh('kgtk filter -p "{tmp_str}" {infile} > {tmp_file}'
                .format(tmp_str=tmp_str, infile=infile, tmp_file=self.tmp_file_path()), self.logger)
        pd_tmp1 = pd.read_csv(self.tmp_file_path(), delimiter='\t')

        # rest of the predicate
        inter_columns = []
        for idx in range(1, len(all_p)):
            p = all_p[idx]
            tmp_str = ';{};'.format(p)
            exec_sh('kgtk filter -p "{tmp_str}" {infile} > {tmp_file}'
                    .format(tmp_str=tmp_str, infile=infile, tmp_file=self.tmp_file_path()), self.logger)
            pd_tmp2 = pd.read_csv(self.tmp_file_path(), delimiter='\t')

            # merge
            if retain_intermediate:
                inter = 'inter_{}'.format(idx)
                pd_tmp1 = pd.merge(pd_tmp1, pd_tmp2, left_on='node2', right_on='node1')[
                    ['node1_x', 'node2_x', 'node2_y'] + inter_columns]
                pd_tmp1 = pd_tmp1.rename(columns={'node1_x': 'node1', 'node2_x': inter, 'node2_y': 'node2'})
                inter_columns.append(inter)
            else:
                pd_tmp1 = pd.merge(pd_tmp1, pd_tmp2, left_on='node2', right_on='node1')[['node1_x', 'node2_y']]
                pd_tmp1 = pd_tmp1.rename(columns={'node1_x': 'node1', 'node2_y': 'node2'})

        return pd_tmp1

    def convert_ttl_to_nt(self, ttl_file, nt_file):
        self.logger.info('converting ttl to nt')
        exec_sh('apache-jena-3.16.0/bin/riot --syntax=ttl --output=nt < {ttl} > {nt}'
                     .format(ttl=ttl_file, nt=nt_file), self.logger)

    def clean_nt(self, nt_file, cleaned_nt_file):
        self.logger.info('cleaning nt')

        self.stat_info = {
            'cluster': 0,
            'prototype': 0,
            'cluster_member': 0,
            'entity': 0,
            'event': 0,
            'relation': 0
        }
        with open(nt_file, 'r') as fin:
            with open(cleaned_nt_file, 'w') as fout:
                for line in fin:
                    line = line.strip()

                    # make cmu id globally unique
                    if config['enable_cmu_gid_patch']:
                        line = line.replace(
                            'http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/',
                            'http://www.lti.cs.cmu.edu/aida/opera/corpora/eval/{}-'.format(self.source)
                        )

                    # statistics
                    m = re_entity.search(line)
                    if m:
                        if m.group(1) == 'Entity':
                            self.stat_info['entity'] += 1
                        elif m.group(1) == 'Event':
                            self.stat_info['event'] += 1
                        elif m.group(1) == 'Relation':
                            self.stat_info['relation'] += 1

                    # statistics
                    # in addition, remove clusters and prototypes
                    m = re_cluster.search(line)
                    if m:
                        if m.group(1) == 'SameAsCluster':
                            self.stat_info['cluster'] += 1
                        elif m.group(1) == 'ClusterMembership':
                            self.stat_info['cluster_member'] += 1
                        elif m.group(1) == 'prototype':
                            self.stat_info['prototype'] += 1
                        continue

                    fout.write(line + '\n')

        self.logger.info(
            'TA1 {} contains {} clusters, {} prototypes, {} cluster members, {} entities, {} events, {} relations'
            .format(self.source,
                    self.stat_info['cluster'], self.stat_info['prototype'], self.stat_info['cluster_member'],
                    self.stat_info['entity'], self.stat_info['event'], self.stat_info['relation']
            ))

    def convert_nt_to_kgtk(self, nt_file, kgtk_file):
        self.logger.info('convert nt to kgtk')
        exec_sh('''kgtk import-ntriples \
      --namespace-file {ns_file} \
      --namespace-id-use-uuid False \
      --newnode-use-uuid False \
      --local-namespace-use-uuid True \
      --local-namespace-prefix {source} \
      --local-namespace-use-uuid False \
      -i {nt_file} > {kgtk_file}'''
        .format(ns_file=config['namespace_file'], source=self.source, nt_file=nt_file, kgtk_file=kgtk_file), self.logger)

    def unreify_kgtk(self, infile, outfile):
        self.logger.info('unreify kgtk')
        exec_sh('kgtk unreify-rdf-statements -i {infile} / sort --columns 1,2 >  {outfile}'
                .format(infile=infile, outfile=outfile), self.logger)

    def create_entity_df(self, kgtk_file, unreified_kgtk_file, output_file, source, ldc_kg, df_wd_fb, kb_to_fb_mapping):
        self.logger.info('create entity df for ' + source)

        ### id
        self.logger.info('creating id')
        exec_sh('kgtk filter -p ";rdf:type;aida:Entity" {kgtk_file} > {tmp_file}'
                .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_entity = pd.read_csv(self.tmp_file_path(), delimiter='\t')
        df_entity = pd.DataFrame({'e': df_entity['node1']})
        if self.stat_info['entity'] != len(df_entity):
            self.logger.error('TA1 has {} entities, TA2 has {} entities'.format(self.stat_info['entity'], len(df_entity)))
        df_entity = df_entity.drop_duplicates().reset_index(drop=True)

        ### name
        self.logger.info('creating name')
        exec_sh('kgtk filter -p ";aida:hasName,aida:textValue;" {kgtk_file} > {tmp_file}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path(1)), self.logger)
        df_name = pd.read_csv(self.tmp_file_path(1), delimiter='\t', error_bad_lines=False, quoting=csv.QUOTE_NONE, doublequote=False).drop(columns=['label']).rename(
            columns={'node1': 'e', 'node2': 'name'})

        def merge_names(names):
            if len(names.index > 0):
                return tuple(names.tolist())
            else:
                return tuple([])

        df_name = df_name.groupby('e')['name'].apply(merge_names).reset_index()

        ### link target
        self.logger.info('creating link target')
        df_tmp1 = self.predicate_path(unreified_kgtk_file, 'aida:link/aida:linkTarget')\
            .rename(columns={'node1': 'e', 'node2': 'target'})
        # remove cluster
        df_target = pd.merge(df_entity, df_tmp1, left_on='e', right_on='e')
        # add confidence
        df_tmp2 = self.predicate_path(unreified_kgtk_file, 'aida:link/aida:confidence/aida:confidenceValue')\
            .rename(columns={'node1': 'e', 'node2': 'cv'})
        df_tmp3 = pd.merge(df_target, df_tmp2, left_on='e', right_on='e')
        df_tmp3 = df_tmp3.groupby('target')['cv'].max().reset_index().rename(columns={'cv': 'score'})
        df_tmp4 = pd.merge(df_target, df_tmp3, left_on='target', right_on='target')
        # merge
        def merge_targets(table):
            if len(table.index) > 0:
                targets = tuple(table['target'].tolist())
                scores = tuple(table['score'].tolist())
                return pd.Series({'target': targets, 'target_score': scores})
            else:
                return pd.Series({'target': tuple([]), 'target_score': tuple([])})

        df_target = df_tmp4.groupby('e')[['target', 'score']].apply(merge_targets).reset_index()
        # expand with labels from ldc kg
        def extract_target_id(t):
            target_id = None
            if t.startswith('REFKB:'):
                target_id = t.split(':')[1]
            elif t.startswith('REFKB'):
                target_id = t[len('REFKB'):]
            return target_id

        def filter_target(df):
            d = df[['e', 'target', 'target_score']].to_dict()
            new_d = {'e': {}, 'target': {}, 'target_score': {}}
            for k, v in d['target'].items():
                if v is None:
                    continue
                new_target = []
                new_target_score = []
                for idx_, tt in enumerate(v):
                    if extract_target_id(tt) is None:  # not valid ID
                        continue
                    new_target.append(tt)
                    new_target_score.append(d['target_score'][k][idx_])
                if len(new_target) > 0:
                    new_d['e'][k] = d['e'][k]
                    new_d['target'][k] = tuple(new_target)
                    new_d['target_score'][k] = tuple(new_target_score)
            return pd.DataFrame(new_d)

        if 'target' not in df_target:
            df_target['target'] = None
        if 'target_score' not in df_target:
            df_target['target_score'] = None
        df_target = filter_target(df_target)

        df_target['target_type'] = None
        df_target['target_name'] = None
        for idx, targets in df_target['target'].iteritems():
            if targets:
                target_type = []
                target_name = []
                for t in targets:
                    # UIUC: REFKB:3634031
                    # BBN: REFKB3643031
                    target_id = extract_target_id(t)
                    if not target_id:
                        target_type.append(None)
                        target_name.append(None)
                        continue

                    data = ldc_kg.get(target_id)
                    if data:
                        target_type.append(data['type'])
                        target_name.append(tuple(data['names']))
                    else:
                        self.logger.warning('Target ID is not in REFKB: {}'.format(target_id))
                        target_type.append(None)
                        target_name.append(None)

                df_target.loc[idx].at['target_type'] = tuple(target_type)
                df_target.loc[idx].at['target_name'] = tuple(target_name)

        # check target again and make sure all target names have value
        # if target name is None, then corresponding target id should be removed
        # if target name is a list of Nones, remove Nones. If if then be an empty list, make it None
        for idx_g, targets in df_target.iterrows():
            target = targets['target']
            target_name = targets['target_name']
            target_score = targets['target_score']

            remove_idx = []
            if target_name:
                for idx, tn in enumerate(target_name):
                    if not tn:
                        remove_idx.append(idx)

                if len(remove_idx) > 0:
                    remove_idx.sort(reverse=True)
                    target = list(target)
                    target_name = list(target_name)
                    target_score = list(target_score)
                    for idx in remove_idx:
                        del target_name[idx]
                        del target[idx]
                        del target_score[idx]

                    if len(target_name) == 0:
                        target_name = None
                        target = None
                        target_score = None
                    else:
                        target = tuple(target)
                        target_name = tuple(target_name)
                        target_score = tuple(target_score)

                    df_target.loc[idx_g]['target'] = target
                    df_target.loc[idx_g]['target_name'] = target_name
                    df_target.loc[idx_g]['target_score'] = target_score
        df_target = df_target.astype('object')

        ### freebase id
        self.logger.info('creating freebase')
        exec_sh('kgtk filter -p ";aida:privateData,aida:jsonContent,aida:system;" {kgtk_file} > {tmp_file}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        exec_sh('kgtk filter -p ";;uiuc:EDL_Freebase" {tmp_file} > {tmp_file1}'
                .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)), self.logger)
        exec_sh('kgtk ifexists --filter-on {tmp_file1} --input-keys node1 --filter-keys node1 -i {tmp_file} | kgtk filter -p ";aida:jsonContent;" > {tmp_file2}'
                .format(tmp_file1=self.tmp_file_path(1), tmp_file=self.tmp_file_path(), tmp_file2=self.tmp_file_path(2)), self.logger)
        exec_sh('head -n 1 {kgtk_file} > {tmp_file3}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file3=self.tmp_file_path(3)), self.logger)
        exec_sh('kgtk filter -p ";aida:privateData;" {kgtk_file} | grep "entity:" >> {tmp_file3}'
                .format(kgtk_file=kgtk_file, tmp_file3=self.tmp_file_path(3)), self.logger)
        df_tmp1 = pd.read_csv(self.tmp_file_path(3), delimiter='\t')
        df_tmp2 = pd.read_csv(self.tmp_file_path(2), delimiter='\t', quoting=csv.QUOTE_NONE, doublequote=False)
        df_fb = pd.merge(df_tmp1, df_tmp2, left_on='node2', right_on='node1')[['node1_x', 'node2_y']].rename(
            columns={'node1_x': 'e', 'node2_y': 'json'})

        def getFBIDs(s):
            if s:
                s = eval(s)
                fbids = []
                avg_scores = []
                max_scores = []
                fbids_json = json.loads(s).get('freebase_link')
                fbids_keys = fbids_json.keys()
                for fbid in fbids_keys:
                    fbids.append(fbid)
                    avg_scores.append(fbids_json.get(fbid).get('average_score'))
                    max_scores.append(fbids_json.get(fbid).get('max_score'))
                return pd.Series(
                    {'fbid': tuple(fbids), 'fbid_score_avg': tuple(avg_scores), 'fbid_score_max': tuple(max_scores)})
            else:
                return pd.Series({'fbid': tuple([]), 'fbid_score_avg': tuple([]), 'fbid_score_max': tuple([])})

        df_fb['fbid'] = None
        df_fb['fbid_score_avg'] = None
        df_fb['fbid_score_max'] = None
        if len(df_fb) > 0:
            df_fb[['fbid', 'fbid_score_avg', 'fbid_score_max']] = df_fb['json'].apply(getFBIDs)
            df_fb = df_fb.drop(columns=['json'])

        def merge_fb(fb):
            fbid = []
            fbid_score_avg = []
            fbid_score_max = []
            for t in fb['fbid'].tolist():
                fbid += list(t)
            for t in fb['fbid_score_avg'].tolist():
                fbid_score_avg += list(t)
            for t in fb['fbid_score_max'].tolist():
                fbid_score_max += list(t)
            return pd.Series(
                {'fbid': tuple(fbid), 'fbid_score_avg': tuple(fbid_score_avg), 'fbid_score_max': tuple(fbid_score_max)})

        if len(df_fb) > 0:
            df_fb = df_fb.groupby('e')[['fbid', 'fbid_score_avg', 'fbid_score_max']].apply(merge_fb).reset_index()

        ### augment fbid
        if kb_to_fb_mapping:
            for idx, target in df_target.iterrows():
                fbid = []
                fbid_score_avg = []
                fbid_score_max = []
                if not target['target']:
                    continue
                for t in target['target']:
                    target_id = extract_target_id(t)
                    if not target_id:
                        continue
                    key = 'REFKB:' + target_id
                    if key in kb_to_fb_mapping:
                        fbs = kb_to_fb_mapping[key].keys()
                        for fb in fbs:
                            fbid.append(fb)
                            fbid_score_avg.append(kb_to_fb_mapping[key][fb])
                            fbid_score_max.append(kb_to_fb_mapping[key][fb])
                df_fb = df_fb.append(pd.Series({'e': target['e'], 'fbid': tuple(fbid),
                    'fbid_score_avg': tuple(fbid_score_avg), 'fbid_score_max': tuple(fbid_score_max)})
                    ,ignore_index=True)
            df_fb.reset_index(drop=True)

        ### embedding vector
        self.logger.info('creating embedding vector')
        exec_sh('kgtk filter -p ";aida:privateData,aida:jsonContent,aida:system;" {kgtk_file} > {tmp_file}'
                .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        exec_sh('kgtk filter -p ";;uiuc:entity_representations" {tmp_file} > {tmp_file1}'
                .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)), self.logger)
        exec_sh('''kgtk ifexists --filter-on {tmp_file1} --input-keys node1 --filter-keys node1 \
    -i {tmp_file} | kgtk filter -p ';aida:jsonContent;' > {tmp_file2}'''
                .format(tmp_file1=self.tmp_file_path(1), tmp_file=self.tmp_file_path(), tmp_file2=self.tmp_file_path(2)), self.logger)
        exec_sh('head -n 1 {kgtk_file} > {tmp_file3}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file3=self.tmp_file_path(3)), self.logger)
        exec_sh('kgtk filter -p ";aida:privateData;" {kgtk_file} | grep "entity:" >> {tmp_file3}'
                .format(kgtk_file=kgtk_file, tmp_file3=self.tmp_file_path(3)), self.logger)
        df_tmp1 = pd.read_csv(self.tmp_file_path(3), delimiter='\t')
        df_tmp2 = pd.read_csv(self.tmp_file_path(2), delimiter='\t', quoting=csv.QUOTE_NONE, doublequote=False)
        df_vector = pd.merge(df_tmp1, df_tmp2, left_on='node2', right_on='node1')[['node1_x', 'node2_y']].rename(
            columns={'node1_x': 'e', 'node2_y': 'vector'})
        df_vector['vector'] = df_vector['vector'].apply(lambda x: json.loads(x))

        ### type
        self.logger.info('creating type')
        exec_sh('kgtk filter -p ";rdf:type;" {kgtk_file} | kgtk filter --invert -p ";;aida:Entity" > {tmp_file}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_tmp1 = pd.read_csv(self.tmp_file_path(), delimiter='\t').rename(columns={'node1': 'e', 'node2': 'type'})
        df_type = pd.merge(df_entity, df_tmp1, left_on='e', right_on='e').drop(columns=['label', 'id'])

        def merge_types(types):
            if len(types.index > 0):
                return tuple(types.tolist())
            else:
                return tuple([])

        df_type = df_type.groupby('e')['type'].apply(merge_types).reset_index()

        ### wikidata
        self.logger.info('creating wikidata')

        def expand(s, multiple=False):
            if s is None:
                return {}
            # expand labels
            result = defaultdict(list) if multiple else {}
            labels = s.split('|')
            for l in labels:
                lang, content = l[-2:], l[1:-4]
                if multiple:
                    result[lang].append(content)
                else:
                    result[lang] = content
            return {k: tuple(result[k]) for k in result.keys()} if multiple else result

        def format_fbid(fbid):
            if not fbid or 'NIL' in fbid: return None
            # .startswith('LDC2015E42:NIL'): return None
            fbid = '/' + fbid.replace('.', '/')
            return fbid

        df_wd = df_fb.copy()
        df_wd['wikidata'] = None
        df_wd['wikidata_label_en'] = None
        df_wd['wikidata_label_ru'] = None
        df_wd['wikidata_label_uk'] = None
        df_wd['wikidata_description_en'] = None
        df_wd['wikidata_description_ru'] = None
        df_wd['wikidata_description_uk'] = None
        df_wd['wikidata_alias_en'] = None
        df_wd['wikidata_alias_ru'] = None
        df_wd['wikidata_alias_uk'] = None

        for idx, line in df_fb.iterrows():
            qnodes = []
            for fbid in line['fbid']:
                fbid = format_fbid(fbid)
                if fbid:
                    qnodes_found = df_wd_fb.loc[df_wd_fb['fbid'] == fbid]['qnode'].values
                    qnodes.append(qnodes_found[0] if len(qnodes_found) > 0 else None)
                else:
                    qnodes.append(None)
            qnodes = tuple(qnodes)
            df_wd.loc[idx].at['wikidata'] = qnodes

            labels = []
            descriptions = []
            aliases = []
            for q in qnodes:
                qnode_value = df_wd_fb.loc[df_wd_fb['qnode'] == q]
                labels.append(expand(qnode_value['label'].values[0]) if qnode_value['label'].size > 0 else {})
                descriptions.append(
                    expand(qnode_value['description'].values[0]) if qnode_value['description'].size > 0 else {})
                aliases.append(
                    expand(qnode_value['alias'].values[0], multiple=True) if qnode_value['alias'].size > 0 else {})

            df_wd.loc[idx].at['wikidata_label_en'] = tuple([l.get('en') for l in labels])
            df_wd.loc[idx].at['wikidata_label_ru'] = tuple([l.get('ru') for l in labels])
            df_wd.loc[idx].at['wikidata_label_uk'] = tuple([l.get('uk') for l in labels])
            df_wd.loc[idx].at['wikidata_description_en'] = tuple([l.get('en') for l in descriptions])
            df_wd.loc[idx].at['wikidata_description_ru'] = tuple([l.get('ru') for l in descriptions])
            df_wd.loc[idx].at['wikidata_description_uk'] = tuple([l.get('uk') for l in descriptions])
            df_wd.loc[idx].at['wikidata_alias_en'] = tuple([l.get('en') for l in aliases])
            df_wd.loc[idx].at['wikidata_alias_ru'] = tuple([l.get('ru') for l in aliases])
            df_wd.loc[idx].at['wikidata_alias_uk'] = tuple([l.get('uk') for l in aliases])

        # ### informative justification
        # self.logger.info('creating informative justification')
        # df_infojust = self.predicate_path(unreified_kgtk_file,
        #     'aida:informativeJustification/aida:confidence/aida:confidenceValue',
        #     retain_intermediate=True) \
        #     .rename(columns={'node1': 'e', 'inter_1': 'informative_justification', 'node2': 'infojust_confidence'}) \
        #     .drop(columns=['inter_2'])
        # df_infojust = pd.merge(df_entity, df_infojust, left_on='e', right_on='e')  # .drop(columns=['label', 'id'])
        #
        # ### justified by
        # self.logger.info('creating justified by')
        # df_just = self.predicate_path(unreified_kgtk_file, 'aida:justifiedBy/aida:confidence/aida:confidenceValue',
        #                          retain_intermediate=True) \
        #     .rename(columns={'node1': 'e', 'inter_1': 'justified_by', 'node2': 'just_confidence'}) \
        #     .drop(columns=['inter_2'])
        # df_just = pd.merge(df_entity, df_just, left_on='e', right_on='e')  # .drop(columns=['label', 'id'])
        #
        # def merge_just(v):
        #     if len(v.index > 0):
        #         confidence = tuple(v['just_confidence'].to_list())
        #         justified_by = tuple(v['justified_by'].to_list())
        #         return pd.Series({'just_confidence': confidence, 'justified_by': justified_by})
        #
        # df_just = df_just.groupby('e')[['just_confidence', 'justified_by']].apply(merge_just).reset_index()

        ### merge
        self.logger.info('merging all dfs to entity df')
        df_entity_complete = pd.merge(df_entity, df_name, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_type, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_target, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_wd, how='left')
        # df_entity_complete = pd.merge(df_entity_complete, df_infojust, how='left')
        # df_entity_complete = pd.merge(df_entity_complete, df_just, how='left')
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
        exec_sh('kgtk filter -p ";rdf:type;aida:Event" {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_event = pd.read_csv(self.tmp_file_path(), delimiter='\t').drop(columns=['node2', 'label'])\
            .rename(columns={'node1': 'e'})
        if self.stat_info['event'] != len(df_event):
            self.logger.error('TA1 has {} events, TA2 has {} events'.format(self.stat_info['event'], len(df_event)))
        df_event = df_event.drop_duplicates().reset_index(drop=True)

        ### type
        self.logger.info('creating type')
        exec_sh('kgtk filter -p ";rdf:type;" {kgtk_file} | kgtk filter --invert -p ";;aida:Event" > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_tmp1 = pd.read_csv(self.tmp_file_path(), delimiter='\t').rename(columns={'node1': 'e', 'node2': 'type'})
        df_event_type = pd.merge(df_event, df_tmp1, left_on='e', right_on='e').drop(columns=['label', 'id'])

        ### name
        self.logger.info('creating name')
        exec_sh('kgtk filter -p ";skos:prefLabel;" {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_event_name = pd.read_csv(self.tmp_file_path(), delimiter='\t', quoting=csv.QUOTE_NONE, doublequote=False)\
            .drop(columns=['label']).rename(columns={'node1': 'e', 'node2': 'name'})

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
        exec_sh('kgtk filter --invert -p ";rdf:type;" {kgtk_file} > {tmp_file}'
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
            exec_sh('kgtk filter -p ";aida:justifiedBy;" {kgtk_file} > {tmp_file}'
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
        exec_sh('kgtk filter -p ";rdf:type;aida:Relation" {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        df_relation = pd.read_csv(self.tmp_file_path(), delimiter='\t').drop(columns=['node2', 'label']).rename(
            columns={'node1': 'e'})
        if self.stat_info['relation'] != len(df_relation):
            self.logger.error('TA1 has {} relations, TA2 has {} relations'.format(self.stat_info['relation'], len(df_relation)))
        df_relation = df_relation.drop_duplicates().reset_index(drop=True)

        ### type
        self.logger.info('creating type')
        exec_sh('kgtk filter -p ";rdf:type;" {kgtk_file} | kgtk filter --invert -p ";;aida:Relation" > {tmp_file}'
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

    def create_relation_role_df(self, kgtk_file, unreified_kgtk_file, output_file, source, entity_file, relation_file):
        self.logger.info('creating relation role df for ' + source)

        # role
        exec_sh('kgtk filter --invert -p ";rdf:type;" {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()), self.logger)
        exec_sh("awk -F'\t' '$2 ~ /^ldcOnt:/' {tmp_file} > {tmp_file1}"
                     .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)), self.logger)
        df_relation_role = pd.DataFrame(columns=['relation', 'role', 'entity'])

        try:
            # entity ids and relation ids
            df_entity = pd.read_hdf(entity_file)['e']
            entity_ids = set([v for v in df_entity.to_dict().values()])
            df_relation = pd.read_hdf(relation_file)['e']
            relation_ids = set([v for v in df_relation.to_dict().values()])

            # read relations
            df_relation_role = pd.read_csv(self.tmp_file_path(1),
                delimiter='\t', index_col=False, header=None, names=['relation', 'role', 'entity', 'statement'])
            df_relation_role['source'] = source

            df_relation_role = df_relation_role.loc[df_relation_role['entity'].isin(entity_ids)]
            df_relation_role = df_relation_role.loc[df_relation_role['relation'].isin(relation_ids)]
            df_relation_role = df_relation_role.drop_duplicates().reset_index(drop=True)

            # justified by
            exec_sh('kgtk filter -p ";aida:justifiedBy;" {kgtk_file} > {tmp_file}'
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


def convert_nan_to_none(df):
    return df.where(pd.notnull(df), None)


def create_wd_to_fb_mapping():
    # only need to run once
    # step 1
    # kgtk import-wikidata \
    #      -i wikidata-20200803-all.json.bz2 \
    #      --node wikidata-20200803-all-nodes.tsv \
    #      --edge wikidata-20200803-all-edges.tsv \
    #      --qual wikidata-20200803-all-qualifiers.tsv \
    #      --explode-values False \
    #      --lang en,ru,uk \
    #      --procs 24 \
    #      |& tee wikidata-20200803-all-import.log
    # step 2
    # kgtk filter -p ';P646;' wikidata-20200803-all-edges.tsv.gz > qnode_to_freebase_20200803.tsv
    # step 3
    # kgtk ifexists --filter-on qnode_to_freebase.tsv --filter-keys node1 \
    # --input-keys id -i wikidata-20200803-all-nodes.tsv.gz > qnode_freebase_20200803.tsv

    # df_wdid_fb = pd.read_csv('qnode_to_free_tsv_file_path', delimiter='\t').drop(columns=['id', 'label', 'rank'])\
    #     .rename(columns={'node1': 'qnode', 'node2': 'fbid'})
    # df_wd_node = pd.read_csv('filtered_qnode_file_path', delimiter='\t').drop(columns={'type'})\
    #     .rename(columns={'id': 'qnode'})
    # df_wd_fb = pd.merge(df_wd_node, df_wdid_fb, left_on='qnode', right_on='qnode')
    # df_wd_fb.to_csv(config['wd_to_fb_file'], index=False)
    pass


def load_ldc_kb():
    kb_names = defaultdict(lambda: {'type': None, 'names': []})

    # entities
    with open(os.path.join(config['ldc_kg_dir'], 'entities.tab')) as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            line = line.strip().split('\t')
            type_, id_, name1 = line[1], line[2], line[3]
            kb_names[id_]['type'] = type_
            kb_names[id_]['names'].append(name1)
            if len(line) >= 5:
                name2 = line[4]
                kb_names[id_]['names'].append(name2)

    # alternative names
    with open(os.path.join(config['ldc_kg_dir'], 'alternate_names.tab')) as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            line = line.strip().split('\t')
            id_, name_ = line[0], line[1]
            kb_names[id_]['names'].append(name_)

    return kb_names


def load_kb_to_fb_mapping():
    mapping = None
    if config['kb_to_fbid_mapping']:
        with open(config['kb_to_fbid_mapping'], 'r') as f:
            mapping = json.load(f)
    return mapping


def load_wd_to_fb_df():
    return convert_nan_to_none(pd.read_csv(config['wd_to_fb_file']))


def worker(source):
    importer = Importer(source=source)
    importer.run()


def process():
    global ldc_kg, df_wd_fb, kb_to_fb_mapping
    logger = get_logger('importer-main')
    logger.info('loading resource')
    ldc_kg = load_ldc_kb()
    df_wd_fb = load_wd_to_fb_df()
    kb_to_fb_mapping = load_kb_to_fb_mapping()

    logger.info('starting multiprocessing mode')
    pp = pyrallel.ParallelProcessor(
        num_of_processor=config['num_of_processor'],
        mapper=worker,
        max_size_per_mapper_queue=config['num_of_processor'] * 2
    )
    pp.start()

    for infile in glob.glob(os.path.join(config['input_dir'], config['run_name'], '*.ttl')):
        source = os.path.basename(infile).split('.')[0]
        pp.add_task(source)
        logger.info('adding task %s' % source)

    pp.task_done()
    pp.join()
    logger.info('all tasks are finished')

    # integrity check
    logger.info('checking file integrity')
    all_ta1_files = set()
    all_ta2_nt_files = set()
    for infile in glob.glob(os.path.join(config['input_dir'], config['run_name'], '*.ttl')):
        source = os.path.basename(infile).split('.')[0]
        all_ta1_files.add(source)
    for infile in glob.glob(os.path.join(config['temp_dir'], config['run_name'], '*/*.cleaned.nt')):
        source = os.path.basename(infile).split('.')[0]
        all_ta2_nt_files.add(source)
        fn = os.path.join(config['temp_dir'], config['run_name'], source, source)
        if not os.path.exists(fn + '.tsv'):
            logger.error('Incorrect KGTK file: {}'.format(source))
        if not os.path.exists(fn + '.entity.h5'):
            logger.error('Incorrect entity df: {}'.format(source))
        if not os.path.exists(fn + '.event.h5'):
            logger.error('Incorrect event df: {}'.format(source))
        if not os.path.exists(fn + '.relation.h5'):
            logger.error('Incorrect relation df: {}'.format(source))
        if not os.path.exists(fn + '.event_role.h5'):
            logger.error('Incorrect event role df: {}'.format(source))
        if not os.path.exists(fn + '.relation_role.h5'):
            logger.error('Incorrect relation role df: {}'.format(source))
    ta2_missing = all_ta1_files - all_ta2_nt_files
    if len(ta2_missing) > 0:
        for source in ta2_missing:
            logger.error('{} has not been parsed'.format(source))
    logger.info('integrity check completed')


def generate_kb_to_wd_mapping(run_name, outfile):
    df_entity = pd.DataFrame()
    for infile in glob.glob(os.path.join(config['temp_dir'], run_name, '*/*.entity.h5')):
        df_entity = df_entity.append(pd.read_hdf(infile))
    df_entity = df_entity.reset_index(drop=True)

    mapping = defaultdict(lambda: defaultdict(float))
    for idx, e in df_entity.iterrows():
        targets = e['target']
        target_scores = e['target_score']
        fbs = e['fbid']
        fb_scores = e['fbid_score_avg']
        if pd.notna(targets) and pd.notna(fbs):
            for i, t in enumerate(targets):
                t_score = target_scores[i]
                for j, fb in enumerate(fbs):
                    fb_score = fb_scores[j]
                    curr_score = 1.0 * t_score * fb_score
                    prev_score = mapping[t].get(fb)
                    if prev_score:
                        mapping[t][fb] = max(curr_score, prev_score)
                    else:
                        mapping[t][fb] = curr_score
    with open(outfile, 'w') as f:
        json.dump(mapping, f)


if __name__ == '__main__':
    argv = sys.argv
    if argv[1] == 'process':
        process()
    elif argv[1] == 'kb_to_wd':
        run_name = argv[2]
        outfile = argv[3]
        generate_kb_to_wd_mapping(outfile)
    elif argv[1] == 'create_namespace':
        outfile = argv[2]

        # pick the file with biggest size
        source = None
        source_size = 0
        for infile in glob.glob(os.path.join(config['input_dir'], config['run_name'], '*.ttl')):
            if not source:
                source = infile
            file_size = os.stat(infile).st_size
            if file_size > source_size:
                source = os.path.basename(infile).split('.')[0]
        im = Importer(source=source)
        im.create_namespace_file(outfile)
