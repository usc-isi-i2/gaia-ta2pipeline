# GAIA TA2 Pipeline

This new pipeline is based on [KGTK](https://github.com/usc-isi-i2/kgtk).


## Docker

Docker build:

```
make docker-build
```

Docker run:

```
make docker-run
```

All configurations can be found in Makefile. Please make changes to them accordingly.

TA2 has three steps: import, clustering, export. 

- To run them all, do `PROD=True python runner.py`. 
- To run importer only, do `PROD=True python importer.py process`.
- To run clusterer only, do `PROD=True python clusterer.py process`.
- To run exporter only, do `PROD=True python exporter.py process`.

For BBN data, please add `-e KB_FBID_MAPPING=/aida/res/kb_to_wd_mapping.json` to enable target id to freebase id mapping while running docker.


## Resource required

All of these are on Goolge shared drives at `GAIA:/gaia-ta2-m36/res` (Please contact me to get access).

- `df_wd_fb.csv`: Wikidata to Freebase mapping.
- `aida-namespaces-*.tsv`: Namespace files.
- `kb_to_wd_mapping.json`: Target id to freebase id mapping.


## Run validator

```
docker run --rm -it \
       -v /tmp/aif_validator:/v \
       --entrypoint /opt/aif-validator/java/target/appassembler/bin/validateAIF \
       nextcenturycorp/aif_validator:latest \
       -o --ont /opt/aif-validator/java/src/main/resources/com/ncc/aif/ontologies/LDCOntologyM36 -d /v
```
