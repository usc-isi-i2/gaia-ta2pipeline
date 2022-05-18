# GAIA TA2 Pipeline

This pipeline is for AIDA Phase 3.


## Build Docker

[AIDA Docker specification](https://nextcentury.atlassian.net/wiki/spaces/AIDAC/pages/2600960001/Phase+3+Docker+Input+Output+Specification)

Docker build:

Place `apache-jena-3.16.0` in the current directory.

```
docker build -t uscisii2/gaia-ta2pipeline .
```

## Run Docker

Environment variables:

- `INPUT`: Input directory.
- `OUTPUT`: Output directory.
- `TEMP` (optional): Temp directory.
- `LOGGING` (optional): It can be `DEBUG`, `INFO` (default), `WARNING`, `ERROR`. 
- `RUN_NAME`: The name of the sub directory to run in `INPUT`.
- `SUBRUN_NAME`: `INTER-TA` or `NIST`. This pipeline searches TA1 ttl files under `${INPUT}/${RUN_NAME}/${SUBRUN_NAME}`.
- `NUM_PROC` : The number of processors to use.
- `NAMESPACE`: The namespace file. Please use different namespace files for different TA1 teams.
- `KGTK_LABELS`: KGTK label file.
- `KGTK_P279`: KGTK P279 file.
- `EXTRACT_MENTION` (optional): If mentions need to be extracted. It's false by default.


Docker run example:

```
docker run --rm -it \
		-e PROD=True \
		-e NUM_PROC=1\
		-e INPUT=/input \
		-e OUTPUT=/output \
		-e RUN_NAME=uiuc \
		-e SUBRUN_NAME=NIST \
		-e TEMP=/output/WORKING \
		-e NAMESPACE=/aida/res/aida-namespaces-base.tsv \
		-e KGTK_LABELS=/aida/res/labels.en.tsv.gz \
		-e KGTK_P279=/aida/res/derived.P279star.tsv.gz \
		-v $$(your host storage)/input:/input:ro \
		-v $$(your host storage)/output:/output \
		-v $$(your host storage)/res:/aida/res \
		uscisii2/ta2
```

> All the resource files are available on CKG's Google shared drive at `GAIA:/gaia-ta2-phase3/res`.