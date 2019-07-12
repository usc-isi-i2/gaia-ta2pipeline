import sys
import os
from gen_dataframe import generate_dataframe
from add_transl_cols import add_trasl_cols
from gen_entity_clusters import gen_entity_clusters_baseline
from gen_event_clusters import gen_event_clusters
kg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gaia-knowledge-graph/update_kg')
sys.path.append(kg_path)
from Updater import Updater
from datetime import datetime


if __name__ == '__main__':
    endpoint = 'http://gaiadev01.isi.edu:7200/repositories'
    repo_src = 'cmu-ta1-0703'
    repo_dst = 'cmu-ta2-0703'
    graph = 'http://www.isi.edu/20190710-001'

    endpoint_src = endpoint + '/' + repo_src
    endpoint_dst = endpoint + '/' + repo_dst
    srcdir = 'store_data/' + repo_src
    outdir = 'store_data/' + repo_dst

    print("Endpoint: ", endpoint)
    print("Src Repository: ", repo_src)
    print("Dst Repository: ", repo_dst)
    print("Graph: ", graph)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    print('Generating dataframe... ', datetime.now().isoformat())
    # generate_dataframe(endpoint_url, outdir)

    print('Augmenting dataframe with translation columns... ', datetime.now().isoformat())
    # add_trasl_cols(outdir + '/entity_all.h5', outdir)

    print('Generating entity clusters... ', datetime.now().isoformat())
    # gen_entity_clusters_baseline(outdir + '/entity_trans_all_filtered.h5', outdir)

    print('Generating event clusters... ', datetime.now().isoformat())
    gen_event_clusters(endpoint_src, outdir)

    print('Insert into GraphDB... ', datetime.now().isoformat())
    up = Updater(endpoint_src, endpoint_dst, repo_dst, outdir, graph, True)
    # up.run_delete_ori()  # don't run this if repo already has named graph
    # up.run_system()  # not needed if repo already has named graph
    up.run_clusters(entity_clusters='clusters-20190710-001.jl')
    up.run_insert_proto()
    up.run_super_edge()
    up.run_inf_just_nt()
    up.run_links_nt()