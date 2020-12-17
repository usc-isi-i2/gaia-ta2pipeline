
docker-build:
	DOCKER_BUILDKIT=1 docker build -t gaiaaida/ta2-full -f Dockerfile ../..

docker-run:
	docker run --rm -it \
		-e INPUT=/input \
		-e OUTPUT=/output \
		-e REPO_KB=/kb \
		-e RUN_NAME=test \
		-e TEMP=/aida/tmp \
		-e NAMESPACE=/aida/res/aida-namespaces-uiuc.tsv \
		-e WD_FB_MAPPING=/aida/res/df_wd_fb.csv \
		-e NUM_PROC=30 \
		-e PARAMS=gaia \
		-v $$(pwd)/../../pipeline_test/input:/input:ro \
		-v $$(pwd)/../../pipeline_test/output:/output \
		-v $$(pwd)/../../pipeline_test/LDC2020E27:/kb \
		-v $$(pwd)/../../pipeline_test/temp:/aida/tmp \
		-v $$(pwd)/../../pipeline_test/res:/aida/res \
		-v $$(pwd):/aida/ta2-full \
		-v $$(pwd)/../pipeline:/aida/ta2-pipeline \
		-v $$(pwd)/../../uke-coref:/aida/uke-coref \
		--entrypoint /bin/bash gaiaaida/ta2-full
