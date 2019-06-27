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

# There should be a directory named store_data/<repo_dst> with these files:
# store_data/<repo_dst>/entity_informative_justification.csv
# store_data/<repo_dst>/entity_links.csv
# store_data/<repo_dst>/entity_clusters.jl

if __name__ == '__main__':

    # inputs
    endpoint = 'http://gaiadev01.isi.edu:7200/repositories'
    repo_src = 'dryrun3ta1-3'
    repo_dst = 'dryrun3ta2-3'
    graph = 'http://www.isi.edu/a_graph_name'

    endpoint_src = endpoint + '/' + repo_src
    endpoint_dst = endpoint + '/' + repo_dst
    outdir = 'store_data/' + repo_dst

    print("Endpoint: ", endpoint)
    print("Src Repository: ", repo_src)
    print("Dst Repository: ", repo_dst)
    print("Graph: ", graph)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    print('Generating event clusters... ', datetime.now().isoformat())
    gen_event_clusters(endpoint_src, outdir)

    print('Insert into GraphDB... ', datetime.now().isoformat())
    up = Updater(endpoint_src, endpoint_dst, repo_src, outdir, graph, True)
    up.run_load_jl()
    # ---- don't run this section if there is already named graph in the dst (TA2) repo ----
    up.run_delete_ori()
    up.run_system()
    # ----------------------------
    up.run_entity_nt()
    up.run_inf_just_nt()
    up.run_links_nt()
    up.run_event_nt()
    up.run_relation_nt()
    up.run_insert_proto()
    up.run_super_edge()
