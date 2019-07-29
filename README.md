## TA2 Workflow
### Setup
1. Import TA1 AIF to GraphDB repository default graph - this will be the source repo (TA2 input)

   * Make sure the repo has read access enabled
   
2. Import TA1 AIF to another GraphDB repository default graph - this will be the destination repo (TA2 output)

   * Make sure the repo has read and write access enabled
   
On the TA2 output repo, the TA1 AIF is imported to the default graph.
Each clustering on the same TA1 output is kept in the same TA2 repo on different named graph

### TA2 Pipeline - see `ta2_runner.py`
1. Generate dataframe from source repo [Currently done from Jupyter Notebook]
2. Add translation columns to  [Currently done from Jupyter Notebook]
3. Clustering [Currently done from Julypter Notebook]
4. Generate singleton event clusters (`gen_event_clusters.py`)
5. Generate AIF (see `gaia-knowledge-graph/update_kg/Updater.py`)

    Triples are created and inserted into the TA2 output repo, specified named graph
    Use ```updater.run_all()``` to insert all the data or run the specific insertions only:
    
      * If this is the first time clustering this repo, delete TA1 clusters in the default graph 
      ```updater.run_delete_ori()```
      * If this is the first time clustering this repo, add TA2 system
      ```updater.run_system()```
      * Insert clusters
      ```upater.run_clusters()```
      * Insert cluster prototypes
      ```upater.run_insert_proto()```
      * Insert superedges
      ```upater.run_super_edge()```
      * Insert informative justifications for clusters
      ```upater.run_inf_just_nt()```
      * Insert links for entity clusters
      ```upater.run_links_nt()```

### TA2 AIF
Export TA2 AIF from GraphDB 