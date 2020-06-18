## TA2 Workflow
### Setup
1. Import TA1 AIF to GraphDB repository default graph - this will be the source repo (TA2 input)

   * Make sure the repo has read access enabled
   
2. Import TA1 AIF to another GraphDB repository default graph - this will be the destination repo (TA2 output)

   * Make sure the repo has read and write access enabled
   
On the TA2 output repo, the TA1 AIF is imported to the default graph.
Each clustering on the same TA1 output is kept in the same TA2 repo on different named graph

### Run TA2 Pipeline (`ta2_runner.py` with parameter file as an argument)
```python ta2_runner.py <input.param>```

#### Sample parameter file:
```
[DEFAULT]
endpoint=http://gaiadev01.isi.edu:7200/repositories  # graphdb endpoint
repo_src=jchen-test-ta1  # repo for TA1 output (read only)
repo_dst=jchen-test-ta2  # repo where TA2 output will be inserted
graph=http://www.isi.edu/002  # graphdb named graph for the clustering
version=002  # string to distinguish different clustering version runs
delete_existing_clusters=False  # True will delete all the existing clusters in TA2 repo (including named graph)
outdir=/nas/home/jchen/store_data/jchen-test-ta2  # a directory where output files will be stored (dataframes, clustering files)
cluster_nb=/lfs1/jupyterhub_data_dir/share/yixiangy/ta2-er.ipynb  # Notebook used to run clustering
```

The pipeline does the followings:
1. Generate dataframe from source repo [Done from Jupyter Notebook]
2. Add translation columns to dataframe [Done from Jupyter Notebook]
3. Clustering [Done from Jupyter Notebook specified in the param]
4. Generate singleton event clusters (`gen_event_clusters.py`)
5. Generate AIF (`gaia-knowledge-graph/update_kg/Updater.py`)

    Triples are created and inserted into the TA2 output repo, specified named graph
    Use ```updater.run_all()``` to insert all the data:
    
      * Delete existing clusters in TA2 repo if `delete_existing_clusters` is true.
        This should only be done once on the repo.
      ```updater.run_delete_ori()```
      * Add TA2 system
      ```updater.run_system()```
      * Insert clusters (entity, event, relation)
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
#### Export TA2 AIF from GraphDB 
To export from the GraphDB GUI
1. select the TA2 output repository
2. Go to Explore --> Graphs overview 
3. Check `The default graph` and the named graph `graph` specified in the parameter file
4. Click on the `Export repository` dropdown and choose `Turtle`
5. Select `Yes` on the warning dialog and AIF will download

### Docker notes
```
# Either build the docker image:
$ docker build . -t gaiaaida/ta2:latest
# or pull the image from dockerhub
$ docker pull gaiaaida/ta2:latest

# Prepare data:
$ mkdir ./input
$ mkdir ./output
$ cp ./test/for_ta2_pipeline_test.zip ./input/
$ cp ./test/kg_test.zip ./input/
$ cd ./input/
$ unzip kg_test.zip 
$ tree
.
├── for_ta2_pipeline_test.zip
├── kg
│   ├── alternate_names.tab
│   ├── entities.tab
│   └── member_states.tab
└── kg_test.zip

1 directory, 5 files
$ cd ..

# Run the docker container:
$ docker run --name=ta2-test -v $(pwd)/input/:/input/ -v $(pwd)/output/:/output/ gaiaaida/ta2:latest

# Observer the results:
$ tree ./output/
./output/
├── export.ttl
└── ta2-test-ta2
    ├── document_001.h5
    ├── entity_valid_001.h5
    ├── entity_with_labels_001.h5
    ├── GenerateDataframe2019.out.ipynb
    ├── relation_001.h5
    └── relation_roles_001.h5

1 directory, 7 files
```
