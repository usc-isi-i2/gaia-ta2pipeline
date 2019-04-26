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
    endpoint = sys.argv[1]  # without repo
    repo = sys.argv[2]
    if len(sys.argv) > 3:
        graph = sys.argv[3]
    else:
        graph = None

    endpoint_url = endpoint + '/' + repo
    outdir = 'store_data/' + repo

    print("Endpoint: ", endpoint)
    print("Repository: ", repo)
    print("Graph: ", graph)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    print('Generating dataframe... ', datetime.now().isoformat())
    generate_dataframe(endpoint_url, outdir)

    print('Augmenting dataframe with translation columns... ', datetime.now().isoformat())
    add_trasl_cols(outdir + '/entity_all.h5', outdir)

    print('Generating entity clusters... ', datetime.now().isoformat())
    gen_entity_clusters_baseline(outdir + '/entity_trans_all.h5', outdir)

    print('Generating event clusters... ', datetime.now().isoformat())
    gen_event_clusters(endpoint_url, outdir)

    print('Insert into GraphDB... ', datetime.now().isoformat())
    up = Updater(endpoint_url, repo, outdir, graph, True)
    up.run_load_jl()
    up.run_delete_ori()
    up.run_system()
    up.run_entity_nt()
    up.run_event_nt()
    up.run_relation_nt()
    up.run_insert_proto()
    up.run_super_edge()
