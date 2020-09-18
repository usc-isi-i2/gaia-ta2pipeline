import subprocess
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


ldc_kg = None
df_wd_fb = None


class Importer(object):

    def __init__(self, source):
        self.source = source
        self.logger = get_logger('importer-' + source)
        self.infile = os.path.join(config['input_dir'], config['run_name'], '{}.ttl'.format(source))
        self.temp_dir = os.path.join(config['temp_dir'], config['run_name'], source)

    def run(self):
        global ldc_kg, df_wd_fb
        os.makedirs(self.temp_dir, exist_ok=True)

        try:

            nt_file = os.path.join(self.temp_dir, '{}.nt'.format(self.source))
            kgtk_file = os.path.join(self.temp_dir, '{}.tsv'.format(self.source))
            unreified_kgtk_file = kgtk_file + '.unreified'
            entity_outfile = os.path.join(self.temp_dir, '{}.entity.h5'.format(self.source))
            event_outfile = os.path.join(self.temp_dir, '{}.event.h5'.format(self.source))
            event_role_outfile = os.path.join(self.temp_dir, '{}.event_role.h5'.format(self.source))
            relation_outfile = os.path.join(self.temp_dir, '{}.relation.h5'.format(self.source))
            relation_role_outfile = os.path.join(self.temp_dir, '{}.relation_role.h5'.format(self.source))

            # self.convert_ttl_to_nt(self.infile, nt_file)
            # self.convert_nt_to_kgtk(nt_file, kgtk_file)
            # self.unreify_kgtk(kgtk_file, unreified_kgtk_file)
            self.create_entity_df(kgtk_file, unreified_kgtk_file, entity_outfile, self.source, ldc_kg, df_wd_fb)
            self.create_event_df(kgtk_file, unreified_kgtk_file, event_outfile, self.source)
            self.create_event_role_df(kgtk_file, unreified_kgtk_file, event_role_outfile, self.source)
            self.create_relation_df(kgtk_file, unreified_kgtk_file, relation_outfile, self.source)
            self.create_relation_role_df(kgtk_file, unreified_kgtk_file, relation_role_outfile, self.source)

        except:
            self.logger.exception('Exception caught in Importer.run()')

        # os.remove(kgtk_file)
        # os.remove(unreified_kgtk_file)
        self.clean_temp_files()

    def exec_sh(self, s):
        self.logger.debug('exec_sh:', s)
        process = subprocess.Popen(s, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout, stderr

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
        self.exec_sh('kgtk filter -p "{tmp_str}" {infile} > {tmp_file}'
                .format(tmp_str=tmp_str, infile=infile, tmp_file=self.tmp_file_path()))
        pd_tmp1 = pd.read_csv(self.tmp_file_path(), delimiter='\t')

        # rest of the predicate
        inter_columns = []
        for idx in range(1, len(all_p)):
            p = all_p[idx]
            tmp_str = ';{};'.format(p)
            self.exec_sh('kgtk filter -p "{tmp_str}" {infile} > {tmp_file}'
                    .format(tmp_str=tmp_str, infile=infile, tmp_file=self.tmp_file_path()))
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
        self.exec_sh('graphy read -c ttl / write -c nt < {ttl} > {nt}'
                     .format(ttl=ttl_file, nt=nt_file))

    def convert_nt_to_kgtk(self, nt_file, kgtk_file):
        self.logger.info('convert nt to kgtk')
        self.exec_sh('''kgtk import-ntriples \
      --namespace-file {ns_file} \
      --namespace-id-use-uuid False \
      --newnode-use-uuid False \
      --local-namespace-use-uuid True \
      --local-namespace-prefix {source} \
      --local-namespace-use-uuid False \
      -i {nt_file} > {kgtk_file}'''
        .format(ns_file=config['namespace_file'], source=self.source, nt_file=nt_file, kgtk_file=kgtk_file))

    def unreify_kgtk(self, infile, outfile):
        self.logger.info('unreify kgtk')
        self.exec_sh('kgtk unreify-rdf-statements -i {infile} / sort --columns 1,2 >  {outfile}'
                .format(infile=infile, outfile=outfile))

    def create_entity_df(self, kgtk_file, unreified_kgtk_file, output_file, source, ldc_kg, df_wd_fb):
        self.logger.info('create entity df for ' + source)

        ### id
        self.logger.info('creating id')
        self.exec_sh('kgtk filter -p ";rdf:type;aida:Entity" {kgtk_file} > {tmp_file}'
                .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()))
        df_entity = pd.read_csv(self.tmp_file_path(), delimiter='\t')
        df_entity = pd.DataFrame({'e': df_entity['node1']})

        ### name
        self.logger.info('creating name')
        self.exec_sh('kgtk filter -p ";aida:hasName,aida:textValue;" {kgtk_file} > {tmp_file}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()))
        df_name = pd.read_csv(self.tmp_file_path(), delimiter='\t').drop(columns=['label']).rename(
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
        df_target['target_type'] = None
        df_target['target_name'] = None
        for idx, targets in df_target['target'].iteritems():
            if targets:
                target_type = []
                target_name = []
                for t in targets:
                    kb_version, target_id = t.split(':')
                    if kb_version != 'REFKB':  # 'LDC2019E43'
                        target_type.append(None)
                        target_name.append(None)
                        continue
                    data = ldc_kg.get(target_id)
                    if data:
                        target_type.append(data['type'])
                        target_name.append(tuple(data['names']))
                    else:
                        target_type.append(None)
                        target_name.append(None)

                df_target.loc[idx].at['target_type'] = tuple(target_type)
                df_target.loc[idx].at['target_name'] = tuple(target_name)

        ### freebase id
        self.logger.info('creating freebase')
        self.exec_sh('kgtk filter -p ";aida:privateData,aida:jsonContent,aida:system;" {kgtk_file} > {tmp_file}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()))
        self.exec_sh('kgtk filter -p ";;rpi:EDL_Freebase" {tmp_file} > {tmp_file1}'
                .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)))
        self.exec_sh('kgtk ifexists --filter-on {tmp_file1} --input-keys node1 --filter-keys node1 -i {tmp_file} | kgtk filter -p ";aida:jsonContent;" > {tmp_file2}'
                .format(tmp_file1=self.tmp_file_path(1), tmp_file=self.tmp_file_path(), tmp_file2=self.tmp_file_path(2)))
        self.exec_sh('head -n 1 {kgtk_file} > {tmp_file3}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file3=self.tmp_file_path(3)))
        self.exec_sh('kgtk filter -p ";aida:privateData;" {kgtk_file} | grep "entity:" >> {tmp_file3}'
                .format(kgtk_file=kgtk_file, tmp_file3=self.tmp_file_path(3)))
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

        ### embedding vector
        self.logger.info('creating embedding vector')
        self.exec_sh('kgtk filter -p ";aida:privateData,aida:jsonContent,aida:system;" {kgtk_file} > {tmp_file}'
                .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()))
        self.exec_sh('kgtk filter -p ";;rpi:entity_representations" {tmp_file} > {tmp_file1}'
                .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)))
        self.exec_sh('''kgtk ifexists --filter-on {tmp_file1} --input-keys node1 --filter-keys node1 \
    -i {tmp_file} | kgtk filter -p ';aida:jsonContent;' > {tmp_file2}'''
                .format(tmp_file1=self.tmp_file_path(1), tmp_file=self.tmp_file_path(), tmp_file2=self.tmp_file_path(2)))
        self.exec_sh('head -n 1 {kgtk_file} > {tmp_file3}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file3=self.tmp_file_path(3)))
        self.exec_sh('kgtk filter -p ";aida:privateData;" {kgtk_file} | grep "entity:" >> {tmp_file3}'
                .format(kgtk_file=kgtk_file, tmp_file3=self.tmp_file_path(3)))
        df_tmp1 = pd.read_csv(self.tmp_file_path(3), delimiter='\t')
        df_tmp2 = pd.read_csv(self.tmp_file_path(2), delimiter='\t', quoting=csv.QUOTE_NONE, doublequote=False)
        df_vector = pd.merge(df_tmp1, df_tmp2, left_on='node2', right_on='node1')[['node1_x', 'node2_y']].rename(
            columns={'node1_x': 'e', 'node2_y': 'vector'})
        df_vector['vector'] = df_vector['vector'].apply(lambda x: json.loads(x))

        ### type
        self.logger.info('creating type')
        self.exec_sh('kgtk filter -p ";rdf:type;" {kgtk_file} | kgtk filter --invert -p ";;aida:Entity" > {tmp_file}'
                .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()))
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

        ### informative justification
        self.logger.info('creating informative justification')
        df_infojust = self.predicate_path(unreified_kgtk_file,
            'aida:informativeJustification/aida:confidence/aida:confidenceValue',
            retain_intermediate=True) \
            .rename(columns={'node1': 'e', 'inter_1': 'informative_justification', 'node2': 'infojust_confidence'}) \
            .drop(columns=['inter_2'])
        df_infojust = pd.merge(df_entity, df_infojust, left_on='e', right_on='e')  # .drop(columns=['label', 'id'])

        ### justified by
        self.logger.info('creating justified by')
        df_just = self.predicate_path(unreified_kgtk_file, 'aida:justifiedBy/aida:confidence/aida:confidenceValue',
                                 retain_intermediate=True) \
            .rename(columns={'node1': 'e', 'inter_1': 'justified_by', 'node2': 'just_confidence'}) \
            .drop(columns=['inter_2'])
        df_just = pd.merge(df_entity, df_just, left_on='e', right_on='e')  # .drop(columns=['label', 'id'])

        def merge_just(v):
            if len(v.index > 0):
                confidence = tuple(v['just_confidence'].to_list())
                justified_by = tuple(v['justified_by'].to_list())
                return pd.Series({'just_confidence': confidence, 'justified_by': justified_by})

        df_just = df_just.groupby('e')[['just_confidence', 'justified_by']].apply(merge_just).reset_index()

        ### merge
        self.logger.info('merging all dfs to entity df')
        df_entity_complete = pd.merge(df_entity, df_name, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_type, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_target, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_wd, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_infojust, how='left')
        df_entity_complete = pd.merge(df_entity_complete, df_just, how='left')
        df_entity_complete['source'] = source

        ### export
        if not self.validate_entity_df(df_entity_complete):
            self.logger.error('Invalid dataframe, please check input data')
            df_entity_complete.to_csv(output_file + '.invalid.csv')
        else:
            self.logger.info('exporting df')
            # df_entity_complete.to_csv(output_file, index=False)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                df_entity_complete.to_hdf(output_file, 'entity', mode='w', format='fixed')
                df_entity_complete.to_csv(output_file + '.csv')

    def validate_entity_df(self, df):
        # for idx, r in df.iterrows():
        #     pass
        if len(df) != len(set(df['e'])):
            self.logger.error('Detected duplicate entities')
            return False

        return True

    def create_event_df(self, kgtk_file, unreified_kgtk_file, output_file, source):
        self.logger.info('creating event df for ' + source)

        ### id
        self.logger.info('creating id')
        self.exec_sh('kgtk filter -p ";rdf:type;aida:Event" {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()))
        df_event = pd.read_csv(self.tmp_file_path(), delimiter='\t').drop(columns=['node2', 'label'])\
            .rename(columns={'node1': 'e'})

        ### type
        self.logger.info('creating type')
        self.exec_sh('kgtk filter -p ";rdf:type;" {kgtk_file} | kgtk filter --invert -p ";;aida:Event" > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()))
        df_tmp1 = pd.read_csv(self.tmp_file_path(), delimiter='\t').rename(columns={'node1': 'e', 'node2': 'type'})
        df_event_type = pd.merge(df_event, df_tmp1, left_on='e', right_on='e').drop(columns=['label', 'id'])

        ### name
        self.logger.info('creating name')
        self.exec_sh('kgtk filter -p ";skos:prefLabel;" {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()))
        df_event_name = pd.read_csv(self.tmp_file_path(), delimiter='\t').drop(columns=['label'])\
            .rename(columns={'node1': 'e', 'node2': 'name'})

        ### informative justification
        self.logger.info('creating informative justification')
        df_event_infojust = self.predicate_path(unreified_kgtk_file,
                                           'aida:informativeJustification/aida:confidence/aida:confidenceValue',
                                           retain_intermediate=True) \
            .rename(columns={'node1': 'e', 'inter_1': 'informative_justification', 'node2': 'infojust_confidence'}) \
            .drop(columns=['inter_2'])
        df_event_infojust = pd.merge(df_event, df_event_infojust, left_on='e',
                                     right_on='e')  # .drop(columns=['label', 'id'])

        ### justified by
        self.logger.info('creating justified by')
        df_event_just = self.predicate_path(unreified_kgtk_file, 'aida:justifiedBy/aida:confidence/aida:confidenceValue',
                                       retain_intermediate=True) \
            .rename(columns={'node1': 'e', 'inter_1': 'justified_by', 'node2': 'just_confidence'}) \
            .drop(columns=['inter_2'])
        df_event_just = pd.merge(df_event, df_event_just, left_on='e', right_on='e')  # .drop(columns=['label', 'id'])

        def merge_event_just(v):
            if len(v.index > 0):
                confidence = tuple(v['just_confidence'].to_list())
                justified_by = tuple(v['justified_by'].to_list())
                return pd.Series({'just_confidence': confidence, 'justified_by': justified_by})

        df_event_just = df_event_just.groupby('e')[['just_confidence', 'justified_by']].apply(
            merge_event_just).reset_index()


        ### merge
        df_event_complete = pd.merge(df_event, df_event_type, how='left')
        df_event_complete = pd.merge(df_event_complete, df_event_infojust, how='left')
        df_event_complete = pd.merge(df_event_complete, df_event_just, how='left')
        df_event_complete['source'] = source

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_event_complete.to_hdf(output_file, 'event', mode='w', format='fixed')
            df_event_complete.to_csv(output_file + '.csv')

    def create_event_role_df(self, kgtk_file, unreified_kgtk_file, output_file, source):
        self.logger.info('creating event role df for ' + source)
        self.exec_sh('kgtk filter --invert -p ";rdf:type;" {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()))
        self.exec_sh("awk -F'\t' '$1 ~ /^event:/ && $2 ~ /^ldcOnt:/ && $3 ~ /^entity:/' {tmp_file} > {tmp_file1}"
                     .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)))
        df_event_role = pd.DataFrame(columns=['event', 'role', 'entity'])
        try:
            df_event_role = pd.read_csv(self.tmp_file_path(1), delimiter='\t').rename(
                columns={'node1': 'entity', 'label': 'role', 'node2': 'event'})
            df_event_role['source'] = source
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
        self.exec_sh('kgtk filter -p ";rdf:type;aida:Relation" {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=kgtk_file, tmp_file=self.tmp_file_path()))
        df_relation = pd.read_csv(self.tmp_file_path(), delimiter='\t').drop(columns=['node2', 'label']).rename(
            columns={'node1': 'e'})

        ### type
        self.logger.info('creating type')
        self.exec_sh('kgtk filter -p ";rdf:type;" {kgtk_file} | kgtk filter --invert -p ";;aida:Relation" > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()))
        df_tmp1 = pd.read_csv(self.tmp_file_path(), delimiter='\t').rename(columns={'node1': 'e', 'node2': 'type'})
        df_relation_type = pd.merge(df_relation, df_tmp1, left_on='e', right_on='e').drop(columns=['label', 'id'])

        ### info just
        self.logger.info('creating informative justification')
        df_relation_infojust = self.predicate_path(unreified_kgtk_file,
                                              'aida:informativeJustification/aida:confidence/aida:confidenceValue',
                                              retain_intermediate=True) \
            .rename(columns={'node1': 'e', 'inter_1': 'informative_justification', 'node2': 'infojust_confidence'}) \
            .drop(columns=['inter_2'])
        df_relation_infojust = pd.merge(df_relation, df_relation_infojust, left_on='e',
                                        right_on='e')  # .drop(columns=['label', 'id'])

        ### justified by
        self.logger.info('creating justified by')
        df_relation_just = self.predicate_path(unreified_kgtk_file, 'aida:justifiedBy/aida:confidence/aida:confidenceValue',
                                          retain_intermediate=True) \
            .rename(columns={'node1': 'e', 'inter_1': 'justified_by', 'node2': 'just_confidence'}) \
            .drop(columns=['inter_2'])
        df_relation_just = pd.merge(df_relation, df_relation_just, left_on='e',
                                    right_on='e')  # .drop(columns=['label', 'id'])

        def merge_relation_just(v):
            if len(v.index > 0):
                confidence = tuple(v['just_confidence'].to_list())
                justified_by = tuple(v['justified_by'].to_list())
                return pd.Series({'just_confidence': confidence, 'justified_by': justified_by})

        df_relation_just = df_relation_just.groupby('e')[['just_confidence', 'justified_by']].apply(
            merge_relation_just).reset_index()

        ### merge
        df_relation_complete = pd.merge(df_relation, df_relation_type, how='left')
        df_relation_complete = pd.merge(df_relation_complete, df_relation_infojust, how='left')
        df_relation_complete = pd.merge(df_relation_complete, df_relation_just, how='left')
        df_relation_complete['source'] = source

        ### export
        self.logger.info('exporting df')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_relation_complete.to_hdf(output_file, 'relation', mode='w', format='fixed')
            df_relation_complete.to_csv(output_file + '.csv')

    def create_relation_role_df(self, kgtk_file, unreified_kgtk_file, output_file, source):
        self.logger.info('creating relation role df for ' + source)
        self.exec_sh('kgtk filter --invert -p ";rdf:type;" {kgtk_file} > {tmp_file}'
                     .format(kgtk_file=unreified_kgtk_file, tmp_file=self.tmp_file_path()))
        self.exec_sh('echo -e "relation\trole\tentity" > {tmp_file1}'.format(tmp_file1=self.tmp_file_path(1)))
        self.exec_sh("awk -F'\t' '$1 ~ /^relation:/ && $2 ~ /^ldcOnt:/ && $3 ~ /^entity:/' {tmp_file} >> {tmp_file1}"
                     .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)))
        self.exec_sh("awk -F'\t' '$1 ~ /^columbia:/ && $2 ~ /^ldcOnt:/ && $3 ~ /^entity:/' {tmp_file} >> {tmp_file1}"
                     .format(tmp_file=self.tmp_file_path(), tmp_file1=self.tmp_file_path(1)))
        df_relation_role = pd.DataFrame(columns=['relation', 'role', 'entity'])
        try:
            df_relation_role = pd.read_csv(self.tmp_file_path(1), delimiter='\t', index_col=False)
            df_relation_role['source'] = source
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


def load_wd_to_fb_df():
    return convert_nan_to_none(pd.read_csv(config['wd_to_fb_file']))


def worker(source):
    importer = Importer(source=source)
    importer.run()


def process():
    global ldc_kg, df_wd_fb
    logger = get_logger('importer-main')
    logger.info('loading resource')
    ldc_kg = load_ldc_kb()
    df_wd_fb = load_wd_to_fb_df()

    logger.info('starting multiprocessing mode')
    pp = pyrallel.ParallelProcessor(
        num_of_processor=config['num_of_processor'],
        mapper=worker,
        max_size_per_mapper_queue=config['num_of_processor'] * 2
    )
    pp.start()

    for infile in glob.glob(os.path.join(config['input_dir'], config['run_name'], 'KC003A6NA.ttl')):
        source = os.path.basename(infile).split('.')[0]
        pp.add_task(source)
        logger.info('adding task %s' % source)

    pp.task_done()
    pp.join()
    logger.info('all tasks are finished')


if __name__ == '__main__':
    # ls | xargs -I {} bash -c 'graphy read -c ttl / write -c nt < {} > {}.nt'
    argv = sys.argv
    if argv[1] == 'process':
        process()
