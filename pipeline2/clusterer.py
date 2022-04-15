import numpy as np
import pandas as pd
import os
import shutil
import io
from copy import deepcopy
import json
import time
import csv
import sys
import random
import math
import string
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import glob
import warnings
from config import config, get_logger
from operator import itemgetter
import requests
import rltk
import gzip
import csv

# from sklearn.cluster import DBSCAN, AgglomerativeClustering
# import graph_tool.all as gt


logger = get_logger('clusterer')
random.seed(2021)


kgtk_p279 = defaultdict(set)


class Cluster(object):

    def __init__(self):
        self.rids = set([])
        self.id_ = None
        self.links = []
        self.link_cvs = []
        self.types = []
        self.type_cvs = []
        # self.asso_claims = []
        # self.claim_semans = []


def load_resource():
    global kgtk_p279
    with gzip.open(config['kgtk_p279'], 'rt') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            if idx > 10:
                break
            kgtk_p279[row['node1']].add(row['node2'])


# def call_semantic_similarity(input_file):
#     file_name = os.path.basename(input_file)
#     files = {
#         'file': (file_name, open(input_file, mode='rb'), 'application/octet-stream')
#     }
#     resp = requests.post(config['kgtk_similarity_url'], files=files, params={'similarity_types': 'all'})
#     if resp.status_code // 100 != 2:
#         logger.error('call_semantic_similarity')
#         return None
#     s = json.loads(resp.json())
#     return pd.DataFrame(s)
#
#
# def call_nearest_neighbor(qnode):
#     resp = requests.get(f'{config["kgtk_nearest_neighbor_url"]}?qnode={qnode}&k={config["kgtk_nearest_neighbor_k"]}')
#     s = resp.json()
#     return s
#
#
# def get_qnode_label(qnode):
#     resp = requests.get(f'{config["kgtk_search_url"]}?q={qnode}&extra_info=true&language=en')
#     s = resp.json()
#     try:
#         s = s[0]['label'][0]
#     except:
#         s = ''
#     return s


# def get_qnode_details_via_es(self, qnodes, labels_only=True):
#     """
#     Modified from https://github.com/usc-isi-i2/kgtk-similarity/blob/main/semantic_similarity/utility.py
#     """
#     embeddings_to_index_field = {
#         "complex": "graph_embedding_complex",
#         "text": "text_embedding",
#         "transe": "graph_embeddings_transe",
#         "class": "class_count"
#     }
#
#     source_fields = ["labels.en"]
#     if not labels_only:
#         source_fields.extend([embeddings_to_index_field[k] for k in embeddings_to_index_field])
#
#     qnodes_dict = {}
#     ids_query = {
#         "_source": source_fields,
#         "query": {
#             "ids": {
#                 "values": qnodes
#             }
#         },
#         "size": len(qnodes)
#     }
#
#     es_search_url = f"{self.config['es_url']}/{self.config['es_index']}/_search"
#     results = requests.post(es_search_url, json=ids_query).json()
#
#     if "hits" in results:
#         hits = results['hits']['hits']
#         for hit in hits:
#             qnode = hit['_id']
#             label = ''
#             if qnode not in qnodes_dict:
#                 qnodes_dict[qnode] = {}
#             _source = hit['_source']
#             for k in embeddings_to_index_field:
#                 if embeddings_to_index_field[k] in _source:
#                     if k == "class":
#                         qnodes_dict[qnode][k] = _source[embeddings_to_index_field[k]]
#                     else:
#
#                         embedding = _source[embeddings_to_index_field[k]]
#                         if isinstance(embedding, str):
#                             # this should be more efficient than doing it by hand:
#                             # TO DO: test with float32 here which should be sufficient
#                             embedding = np.fromstring(embedding, dtype=float, sep=',')
#                         else:
#                             embedding = np.array([float(x) for x in embedding])
#                         qnodes_dict[qnode][k] = embedding
#             if 'labels' in _source:
#                 _labels = _source['labels']
#                 if 'en' in _labels:
#                     label = _labels['en'][0]
#                 qnodes_dict[qnode]['label'] = label
#
#     return qnodes_dict


def process():

    # load_resource()

    df_entity = pd.DataFrame()
    df_event = pd.DataFrame()
    df_relation = pd.DataFrame()
    df_role = pd.DataFrame()

    logger.info('loading entity dataframes')
    for infile in glob.glob(os.path.join(config['temp_dir'], config['run_name'], config["subrun_name"], '*/*.entity.h5')):
        source = os.path.basename(infile).split('.')[0]
        # entity
        df_entity = df_entity.append(pd.read_hdf(infile))
        # event
        event_file = infile[:-len('entity.h5')] + 'event.h5'
        df_event = df_event.append(pd.read_hdf(event_file))
        # relation
        relation_file = infile[:-len('entity.h5')] + 'relation.h5'
        df_relation = df_relation.append(pd.read_hdf(relation_file))
        # role
        role_file = infile[:-len('entity.h5')] + 'role.h5'
        df_role = df_role.append(pd.read_hdf(role_file))

    logger.info(f'Read in {len(df_entity)} entities, {len(df_event)} events, {len(df_relation)} relations, {len(df_role)} roles')
    df_entity = df_entity.drop_duplicates(subset=['e'], keep='last')  # cmu data has cross document entities, only keep one
    df_entity = df_entity.reset_index(drop=True)
    df_entity = df_entity[df_entity['type'].notnull()]  # drop the entity with no type
    # df_entity['type'] = df_entity['type'].apply(lambda x: x[0])  # only pick the first type (compatible with old pipeline)
    # df_entity['type_cv'] = df_entity['type_cv'].apply(lambda x: x[0])
    df_entity_ori = df_entity.copy()
    df_event = df_event.drop_duplicates(subset=['e'], keep='last').reset_index(drop=True)
    df_event['proto'] = df_event['proto'].apply(lambda x: x[0])
    df_event['cluster'] = df_event['cluster'].apply(lambda x: x[0])
    df_relation = df_relation.drop_duplicates(subset=['e'], keep='last').reset_index(drop=True)
    df_relation['proto'] = df_relation['proto'].apply(lambda x: x[0])
    df_relation['cluster'] = df_relation['cluster'].apply(lambda x: x[0])
    # different justifications make multiple rows in df_role
    df_role = df_role.drop_duplicates(subset=['e1', 'e1_type', 'e2', 'e2_type', 'role', 'just'], keep='last').reset_index(drop=True)
    logger.info(f'After deduplication: {len(df_entity)} entities, {len(df_event)} events, {len(df_relation)} relations, {len(df_role)} roles')

    logger.info('exporting clusters')
    df_entity_cluster = df_entity_ori.copy()
    df_entity_cluster['cluster'] = None
    df_entity_cluster['synthetic'] = False
    df_entity_cluster['cluster_member_cv'] = None

    entities = {row['e']: row for row in df_entity_cluster.to_dict(orient='records')}
    unclustered_entities = set(df_entity_cluster['e'].to_list())
    clusters = []

    # select the best link
    entity_best_link = {}
    link_to_entity_mapping = defaultdict(set)
    for idx, row in df_entity_cluster.iterrows():
        e = row['e']
        links = row['link']
        link_cvs = row['link_cv']

        # skip entities which have no links
        if pd.isna(links) or pd.isna(link_cvs):
            continue

        best_link, best_cv = None, 0
        for link, cv in zip(links, link_cvs):
            if cv >= best_cv:
                best_cv = cv
                best_link = link
        entity_best_link[e] = (best_link, best_cv)
        link_to_entity_mapping[best_link].add(e)

    # select the best type
    entity_best_type = {}
    type_to_entity_mapping = defaultdict(set)
    for idx, row in df_entity_cluster.iterrows():
        e = row['e']
        types = row['type']
        type_cvs = row['type_cv']

        if pd.isna(types) or pd.isna(type_cvs):
            continue

        best_type, best_cv = None, 0
        for type_, cv in zip(types, type_cvs):
            if cv >= best_cv:
                best_cv = cv
                best_type = type_
        entity_best_type[e] = (best_type, best_cv)
        type_to_entity_mapping[best_type].add(e)

    # # others
    # entity_to_asso_claims_mapping = {}
    # entity_to_claim_semans_mapping = {}
    # for idx, row in df_entity_cluster.iterrows():
    #     if not pd.isna(row['asso_claim']):
    #         entity_to_asso_claims_mapping[row['e']] = row['asso_claim']
    #     if not pd.isna(row['claim_seman']):
    #         entity_to_claim_semans_mapping[row['e']] = row['claim_seman']

    # cluster on links (only one link)
    for es in link_to_entity_mapping.values():
        es = list(es)
        c = Cluster()
        c.rids = es
        for e in es:
            unclustered_entities.remove(e)

        # populate link
        c.links = [entity_best_link[es[0]][0]]
        c.link_cvs = [np.mean([entity_best_link[e][1] for e in es])]

        # populate types
        type_cvs = defaultdict(list)
        for e in es:
            best_type = entity_best_type.get(e)
            if not best_type:
                continue
            type_cvs[best_type[0]].append(best_type[1])
        for t in type_cvs.keys():
            type_cvs[t] = np.mean(type_cvs[t])
        type_cvs = [(t, cv) for t, cv in type_cvs.items()]

        if type_cvs:
            c.types = [t[0] for t in type_cvs]
            c.type_cvs = [t[1] for t in type_cvs]

        # # populate asso claims
        # asso_claims = set()
        # for e in es:
        #     for claim in entity_to_asso_claims_mapping.get(e, []):
        #         asso_claims.add(claim)
        # c.asso_claims = list(asso_claims)
        #
        # # populate claim semans
        # claim_semans = set()
        # for e in es:
        #     for claim in entity_to_claim_semans_mapping.get(e, []):
        #         claim_semans.add(claim)
        # c.claim_semans = list(claim_semans)

        clusters.append(c)

    # # cluster on types (only one type)
    # for type_, es in type_to_entity_mapping.items():
    #     es = filter(lambda x: x in unclustered_entities, es)
    #     es = list(es)
    #     if len(es) == 0:
    #         continue
    #
    #     c = Cluster()
    #     c.rids = es
    #     for e in es:
    #         unclustered_entities.remove(e)
    #
    #     c.types = [entity_best_type[es[0]][0]]
    #     c.type_cvs = [np.mean([entity_best_type[e][1] for e in es])]
    #
    #     clusters.append(c)

    # rest of the entities
    for e in unclustered_entities:
        c = Cluster()
        c.rids = [e]
        clusters.append(c)
    del unclustered_entities

    # entity_type_set = set(df_entity['type'].to_list())
    # entity_types = list(entity_type_set)
    # logger.info(f'entity types {len(entity_types)}')
    # # cache entity labels
    # entity_type_labels = {}
    # entity_type_label_cache_file = os.path.join(config['temp_dir'], config['run_name'], 'entity_type_labels.jl')
    # if not os.path.exists(entity_type_label_cache_file):
    #     logger.info(f'caching entity label...')
    #     for qnode in tqdm(entity_type_set):
    #         entity_type_labels[qnode] = get_qnode_label(qnode)
    #     with open(entity_type_label_cache_file, 'w') as f:
    #         json.dump(entity_type_labels, f)
    # else:
    #     with open(entity_type_label_cache_file, 'r') as f:
    #         entity_type_labels = json.load(f)
    # df_entity_cluster['type_label'] = df_entity_cluster['type'].apply(lambda x: entity_type_labels.get(x))
    #
    # # blocking
    # logger.info('Fetching nearest neighbors')
    # nearest_neighbors = {}
    # nearest_neighbor_cache_file = os.path.join(config['temp_dir'], config['run_name'], 'nearest_neighbor_cache.jl')
    # if not os.path.exists(nearest_neighbor_cache_file):
    #     logger.info(f'caching nearest neighbors...')
    #     with open(nearest_neighbor_cache_file, 'w') as f:
    #         for qnode in tqdm(entity_types):
    #             r = call_nearest_neighbor(qnode)
    #             nearest_neighbors[qnode] = r
    #             f.write(json.dumps({qnode: r}) + '\n')
    # else:
    #     with open(nearest_neighbor_cache_file, 'r') as f:
    #         for line in f:
    #             obj = json.loads(line)
    #             k = list(obj.keys())[0]
    #             v = list(obj.values())[0]
    #             nearest_neighbors[k] = v
    #
    # nearest_neighbor_block = defaultdict(set)
    # for k, v in nearest_neighbors.items():
    #     for vv in v:
    #         nearest_neighbor_block[k].add(vv['qnode'])
    #     # only retain the nodes that are in input
    #     nearest_neighbor_block[k] = nearest_neighbor_block[k] & entity_type_set
    # # with open(os.path.join(config['temp_dir'], config['run_name'], 'blocks.json'), 'w') as f:
    # #     json.dump(nearest_neighbor_block, f)
    #
    # # compute pairwise similarity
    # logger.info('Computing pairwise similarity')
    # df_entity_similarity = None
    # entity_similarity_cache_file = os.path.join(config['temp_dir'], config['run_name'], 'entity_similarity_cache.tsv')
    #
    # if not os.path.exists(entity_similarity_cache_file):
    #     logger.info(f'caching similarity...')
    #     tmp_entity_file = os.path.join(config['temp_dir'], config['run_name'], 'entity_similarity_input.tsv')
    #
    #     # full_pairs = list(combinations(entity_types, r=2))
    #     cand_pairs = set()
    #     for q1 in entity_types:
    #         for q2 in nearest_neighbor_block[q1]:
    #             if q1 > q2:
    #                 q1, q2 = q2, q1
    #             cand_pairs.add((q1, q2))
    #     cand_pairs = list(cand_pairs)
    #
    #     chunk_size = config['kgtk_similarity_chunk_size']
    #     for i in tqdm(range(math.ceil(len(cand_pairs) / chunk_size))):
    #         si = i * chunk_size
    #         ei = (i + 1) * chunk_size
    #
    #         # logger.info(f'Requesting similarity for chunk {i}')
    #         with open(tmp_entity_file, 'w') as f:
    #             f.write('q1\tq2\n')
    #             for t1, t2 in cand_pairs[si:ei]:
    #                 f.write(f'{t1}\t{t2}\n')
    #
    #         df_entity_similarity_tmp = call_semantic_similarity(tmp_entity_file)
    #         if df_entity_similarity_tmp is None:
    #             shutil.copyfile(tmp_entity_file, f'{tmp_entity_file}.err.{time.time()}')
    #             continue
    #         # df_entity_similarity_tmp.to_csv(entity_similarity_cache_file + f'.{i}', index=False, sep='\t')
    #         if df_entity_similarity is None:
    #             df_entity_similarity = df_entity_similarity_tmp.copy()
    #         else:
    #             df_entity_similarity = pd.concat([df_entity_similarity, df_entity_similarity_tmp], axis=0, ignore_index=True)
    #     df_entity_similarity.to_csv(entity_similarity_cache_file, index=False, sep='\t')
    #
    # else:
    #     df_entity_similarity = pd.read_csv(entity_similarity_cache_file, sep='\t')
    # # print(df_entity_similarity)
    #
    # # clustering
    # logger.info('Clustering')
    # # https://stackoverflow.com/questions/18909096/clustering-given-pairwise-distances-with-unknown-cluster-number
    # entity_type_to_id_mapping = {t: idx for idx, t in enumerate(entity_types)}
    # sim_matrix = np.full(shape=(len(entity_types), len(entity_types)), fill_value=100)  # the default distance is 100
    # for _, row in df_entity_similarity[['q1', 'q2', 'topsim']].iterrows():
    #     idx1 = entity_type_to_id_mapping[row['q1']]
    #     idx2 = entity_type_to_id_mapping[row['q2']]
    #     # x100 because the range is in [0,1]
    #     sim_matrix[idx1][idx2] = (1 - row['topsim']) * 100
    #     sim_matrix[idx2][idx1] = (1 - row['topsim']) * 100
    #
    # # clusters = DBSCAN(eps=0.45, metric='precomputed').fit(sim_matrix)
    # # labels = clusters.labels_
    # # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # # print(n_clusters_)
    # clusters = AgglomerativeClustering(n_clusters=None,
    #                                    distance_threshold=50,
    #                                    linkage='average',
    #                                    affinity='precomputed').fit(sim_matrix)
    # clusters = clusters.labels_
    # cluster_ids = [''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
    #                for _ in range(max(clusters)+1)]
    # logger.info(f'Found {max(clusters)+1} clusters')
    # # print(clusters)

    # # community detection (not used for now)
    # g = gt.Graph(directed=False)
    # g.add_vertex(len(entity_types))
    # for _, row in df_entity_similarity[['q1', 'q2', 'topsim']].iterrows():
    #     idx1 = entity_type_to_id_mapping[row['q1']]
    #     idx2 = entity_type_to_id_mapping[row['q2']]
    #     g.add_edge(g.vertex(idx1), g.vertex(idx2))
    # # state = gt.minimize_blockmodel_dl(g)
    # # state.draw(output=os.path.join(config['temp_dir'], config['run_name'], 'clusters_gt_block.svg'))
    # # print(list(state.get_blocks()))
    # state = gt.minimize_nested_blockmodel_dl(g)
    # for l in state.levels:
    #     print(list(l.get_blocks()))


    # # cluster numeric id to string id
    # qnode_cluster_mapping = {}
    # for t, c in zip(entity_types, clusters):
    #     qnode_cluster_mapping[t] = cluster_ids[c]

    cid_to_cluster = defaultdict(set)
    # generate cluster id
    for c in clusters:
        cid = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
        c.id_ = cid
        cid_to_cluster[cid] = c
        for e in c.rids:
            entities[e]['cluster'] = cid

    # prepare output
    # assign string id to entities
    logger.info('updating cluster info for each entity')
    for idx, row in df_entity_cluster[['e']].iterrows():
        df_entity_cluster.at[idx, 'cluster'] = 'gaia:entity/cluster/' + entities[row['e']]['cluster']
        df_entity_cluster.at[idx, 'cluster_member_cv'] = 1.0
    # print(len(df_entity_cluster))

    # # elect link
    # cluster_links = {}
    # for cid in cluster_ids:
    #
    #     link_cv_agg = defaultdict(list)
    #     # aggregate cvs
    #     for idx, row in df_entity_cluster[df_entity_cluster['cluster'] == 'gaia:entity/cluster/' + cid].iterrows():
    #         links = row['link']
    #         cvs = row['link_cv']
    #         for link, cv in zip(links, cvs):
    #             link_cv_agg[link].append(cv)
    #     link_cv_agg = {k: np.mean(v) for k, v in link_cv_agg.items()}
    #     cluster_links[cid] = link_cv_agg

    logger.info('creating prototypes')
    # df_entity_prototype = df_entity_cluster.groupby(['cluster']).head(1).reset_index(drop=True)
    prototype_dict = {'e': [], 'cluster': [], 'synthetic': [], 'link': [],
                      'link_cv': [], 'type': [], 'type_cv': []}
    for c in clusters:
        cid = c.id_
        prototype_dict['e'].append(f'gaia:entity/prototype/{cid}')
        prototype_dict['cluster'].append(f'gaia:entity/cluster/{cid}')
        prototype_dict['synthetic'].append(True)
        # links = [(link, cv) for link, cv in cluster_links[cid].items()]
        prototype_dict['link'].append(tuple(c.links))
        prototype_dict['link_cv'].append(tuple(c.link_cvs))
        prototype_dict['type'].append(tuple(c.types))
        prototype_dict['type_cv'].append(tuple(c.type_cvs))
    df_entity_prototype = pd.DataFrame.from_dict(prototype_dict)

    # print(prototype_dict)
    # print(df_entity_prototype.shape)
    # print(df_entity_prototype.drop_duplicates().shape)
    # cluster_prefixes = {row['cluster']: row['e'] for _, row in df_entity_prototype[['e', 'cluster']].iterrows()}
    # # need to make prototype different from real entity in order to satisfy the validator
    # df_entity_prototype['e'] = df_entity_prototype['e'].apply(lambda x: f'{x}-prototype')

    logger.info('appending dataframes')
    df_complete_entity_clusters = df_entity_cluster.append(df_entity_prototype).reset_index(drop=True)
    # update cluster string id to uri
    # df_complete_entity_clusters['cluster'] = df_complete_entity_clusters['cluster']\
    #     .apply(lambda x: f'{cluster_prefixes[x]}-cluster-{x}')

    logger.info('writing to disk')
    entity_cluster_output_file = os.path.join(config['temp_dir'], config['run_name'], config["subrun_name"], 'entity_cluster')
    # event_output_file = os.path.join(config['temp_dir'], config['run_name'], 'event')
    # relation_output_file = os.path.join(config['temp_dir'], config['run_name'], 'relation')
    # role_output_file = os.path.join(config['temp_dir'], config['run_name'], 'role')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df_complete_entity_clusters.to_hdf(entity_cluster_output_file + '.h5', 'entity', mode='w', format='fixed')
        df_complete_entity_clusters.to_csv(entity_cluster_output_file + '.h5.csv')

    ### construct super edge
    logger.info('constructing super edge')

    entity_to_proto = {}
    for c in clusters:
        cid = c.id_
        for rid in c.rids:
            entity_to_proto[rid] = f'gaia:entity/prototype/{cid}'
    event_to_proto = {}
    for _, row in df_event[['e', 'proto']].iterrows():
        event_to_proto[row['e']] = row['proto']
    relation_to_proto = {}
    for _, row in df_relation[['e', 'proto']].iterrows():
        relation_to_proto[row['e']] = row['proto']

    # merge edges to be super edges
    super_edge_merged = defaultdict(
        lambda: {'just': set([]), 'cv': 0.0, 'proto1': proto1, 'proto2': proto2, 'role': role})
    for _, row in df_role.iterrows():
        e1 = row['e1']
        e2 = row['e2']
        e1_type = row['e1_type']
        e2_type = row['e2_type']
        role = row['role']
        cv = row['cv']
        just = row['just']

        proto1 = None
        if e1_type == 'aida:Event':
            proto1 = event_to_proto[e1]
        elif e1_type == 'aida:Relation':
            proto1 = relation_to_proto[e1]
        elif e1_type == 'aida:Entity':
            proto1 = entity_to_proto[e1]
        else:
            logger.error(f'Unknown type1 {e1_type} for prototype1 {proto1} while creating the super edge')

        proto2 = None
        if e2_type == 'aida:Event':
            proto2 = event_to_proto[e2]
        elif e2_type == 'aida:Relation':
            proto2 = relation_to_proto[e2]
        elif e2_type == 'aida:Entity':
            proto2 = entity_to_proto[e2]
        else:
            logger.error(f'Unknown type2 {e2_type} for prototype2 {proto2} while creating the super edge')

        k = f'{proto1}-{proto2}-{role}'  # key for merging
        super_edge_merged[k]['just'].add(just)
        super_edge_merged[k]['cv'] = max(super_edge_merged[k]['cv'], cv)

    # construct dataframe
    super_edges = {'proto1': [], 'proto2': [], 'role': [], 'cv': [], 'just': []}
    for _, v in super_edge_merged.items():
        super_edges['proto1'].append(v['proto1'])
        super_edges['proto2'].append(v['proto2'])
        super_edges['role'].append(v['role'])
        super_edges['cv'].append(v['cv'])
        super_edges['just'].append(tuple(v['just']))
    df_super_edge = pd.DataFrame.from_dict(super_edges)

    super_edge_output_file = os.path.join(config['temp_dir'], config['run_name'], config["subrun_name"], 'super_edge')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df_super_edge.to_hdf(super_edge_output_file + '.h5', 'super_edge', mode='w', format='fixed')
        df_super_edge.to_csv(super_edge_output_file + '.h5.csv')


    # viz
    # qnode_labels = {}
    # for _, row in df_entity_similarity.iterrows():
    #     qnode_labels[row['q1']] = row['q1_label']
    #     qnode_labels[row['q2']] = row['q2_label']
    # for k, v in nearest_neighbors.items():
    #     for vv in v:
    #         qnode_labels[vv['qnode']] = vv['label']
    #
    # def format_meta(meta):
    #     return f"ID: {meta['e']}<br>type: {meta['type']}<br>type label: {meta['type_label']}<br>Cluster: {meta['cluster']}"
    #
    # clusters_for_viz = {'nodes': [], 'links': []}
    # for c in set(qnode_cluster_mapping.values()):
    #     clusters_for_viz['nodes'].append({'id': c, 'val': 3})
    # for _, row in df_entity_cluster.iterrows():
    #     clusters_for_viz['nodes'].append({'id': row['e'], 'meta': format_meta(row), 'val': 1, 'group': row['type']})
    #     clusters_for_viz['links'].append({'source': row['e'], 'target': row['cluster'], 'val': 1})
    # # print(clusters_for_viz)
    # with open(os.path.join(config['temp_dir'], config['run_name'], 'clusters_for_viz.json'), 'w') as f:
    #     json.dump(clusters_for_viz, f)


    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore')
    #     # event
    #     event_cluster_output_file = os.path.join(config['temp_dir'], config['run_name'], 'event_cluster.h5')
    #     df_event.to_hdf(event_cluster_output_file, 'event')
    #     event_role_output_file = os.path.join(config['temp_dir'], config['run_name'], 'event_role.h5')
    #     df_event_role_se.to_hdf(event_role_output_file, 'event_role')
    #     df_event_role_se.to_csv(event_role_output_file + '.csv')
    #     # relation
    #     relation_cluster_output_file = os.path.join(config['temp_dir'], config['run_name'], 'relation_cluster.h5')
    #     df_relation.to_hdf(relation_cluster_output_file, 'relation')
    #     relation_role_output_file = os.path.join(config['temp_dir'], config['run_name'], 'relation_role.h5')
    #     df_relation_role_se.to_hdf(relation_role_output_file, 'relation_role', mode='w', format='fixed')
    #     df_relation_role_se.to_csv(relation_role_output_file + '.csv')


if __name__ == '__main__':

    argv = sys.argv
    if argv[1] == 'process':
        process()
