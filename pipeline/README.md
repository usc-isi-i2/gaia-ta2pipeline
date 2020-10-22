# GAIA TA2 Pipeline

This new pipeline is based on [KGTK](https://github.com/usc-isi-i2/kgtk).


## Docker

Docker build:

```
make docker-build
```

Environment variables:

- `INPUT`: Input directory.
- `OUTPUT`: Output directory.
- `TEMP` (optional): Temp directory.
- `REPO_KB`: The path of LDC REFKB.
- `RUN_NAME`: The name of the sub directory to run in `INPUT`. TA1 ttl files should be placed here.
- `NUM_PROC` : The number of processors to use.
- `NAMESPACE`: The namespace file. Please use different namespace files for different TA1 teams.
- `WD_FB_MAPPING`: Wikidata to Freebase mapping.
- `KB_FBID_MAPPING` (optional): The path of the REFKB to Wikidata mapping file. This needs to be set for non-UIUC TA1 data.

Docker run:

```
make docker-run-{ta1 team}
```

All configurations can be found in Makefile. *Please make changes to them accordingly*.

TA2 has three steps: import, clustering, export. 
If you wish to run them step-by-step, please add `--entrypoint /bin/bash` while running the container (see `docker-run-debug` in Makefile).

- To run them all, do `PROD=True python runner.py`. 
- To run importer only, do `PROD=True python importer.py process`.
- To run clusterer only, do `PROD=True python clusterer.py process`.
- To run exporter only, do `PROD=True python exporter.py process`.

## Resource required

All of these are on Goolge shared drives at `GAIA:/gaia-ta2-m36/res` (Please contact me to get access).

- `df_wd_fb.csv`: Wikidata to Freebase mapping.
- `aida-namespaces-{ta1 team}.tsv`: Namespace files.
- `kb_to_wd_mapping.json`: REFKB id to freebase id mapping.


## Run validator

```
docker run --rm -it \
       -v /tmp/aif_validator:/v \
       --entrypoint /opt/aif-validator/java/target/appassembler/bin/validateAIF \
       nextcenturycorp/aif_validator:latest \
       -o --ont /opt/aif-validator/java/src/main/resources/com/ncc/aif/ontologies/LDCOntologyM36 -d /v
```
