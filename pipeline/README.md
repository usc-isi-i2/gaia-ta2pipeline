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


## Resource required

- `df_wd_fb.csv`: Wikidata to Freebase mapping.
- `aida-namespaces.tsv`: Namespace file.


## Run validator

```
docker run --rm -it -v /tmp/aif_validator:/v -e VALIDATION_HOME=/opt/aif-validator -e VALIDATION_FLAGS=--TA2 -e TARGET_TO_VALIDATE=/v --name aifvalidator nextcenturycorp/aif_validator
```
