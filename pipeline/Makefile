docker-build:
	docker build -t gaiaaida/ta2 .

docker-clean:
	docker rmi -f $$(docker images -f "dangling=true" -q)

docker-run-uiuc:
	docker run --rm -it \
		-e INPUT=/input \
		-e OUTPUT=/output \
		-e REPO_KB=/kb \
		-e RUN_NAME=uiuc \
		-e TEMP=/aida/tmp \
		-e NAMESPACE=/aida/res/aida-namespaces-uiuc.tsv \
		-e WD_FB_MAPPING=/aida/res/df_wd_fb.csv \
		-e NUM_PROC=30 \
		-v $$(pwd)/../pipeline_test/input:/input:ro \
		-v $$(pwd)/../pipeline_test/output:/output \
		-v $$(pwd)/../pipeline_test/LDC2020E27:/kb \
		-v $$(pwd)/../pipeline_test/temp:/aida/tmp \
		-v $$(pwd)/../pipeline_test/res:/aida/res \
		gaiaaida/ta2

docker-run-bbn:
	docker run --rm -it \
		-e INPUT=/input \
		-e OUTPUT=/output \
		-e REPO_KB=/kb \
		-e RUN_NAME=bbn \
		-e TEMP=/aida/tmp \
		-e NAMESPACE=/aida/res/aida-namespaces-bbn.tsv \
		-e WD_FB_MAPPING=/aida/res/df_wd_fb.csv \
		-e KB_FBID_MAPPING=/aida/res/kb_to_wd_mapping.json\
		-e NUM_PROC=30 \
		-v $$(pwd)/../../pipeline_test/input:/input:ro \
		-v $$(pwd)/../../pipeline_test/output:/output \
		-v $$(pwd)/../../pipeline_test/LDC2020E27:/kb \
		-v $$(pwd)/../../pipeline_test/temp:/aida/tmp \
		-v $$(pwd)/../../pipeline_test/res:/aida/res \
		gaiaaida/ta2

docker-run-cmu:
	docker run --rm -it \
		-e INPUT=/input \
		-e OUTPUT=/output \
		-e REPO_KB=/kb \
		-e RUN_NAME=cmu \
		-e TEMP=/aida/tmp \
		-e NAMESPACE=/aida/res/aida-namespaces-cmu.tsv \
		-e WD_FB_MAPPING=/aida/res/df_wd_fb.csv \
		-e KB_FBID_MAPPING=/aida/res/kb_to_wd_mapping.json\
		-e NUM_PROC=30 \
		-v $$(pwd)/../../pipeline_test/input:/input:ro \
		-v $$(pwd)/../../pipeline_test/output:/output \
		-v $$(pwd)/../../pipeline_test/LDC2020E27:/kb \
		-v $$(pwd)/../../pipeline_test/temp:/aida/tmp \
		-v $$(pwd)/../../pipeline_test/res:/aida/res \
		gaiaaida/ta2

docker-run-debug:
	docker run --rm -it \
		-e INPUT=/input \
		-e OUTPUT=/output \
		-e REPO_KB=/kb \
		-e RUN_NAME=test \
		-e TEMP=/aida/tmp \
		-e NAMESPACE=/aida/res/aida-namespaces-uiuc.tsv \
		-e WD_FB_MAPPING=/aida/res/df_wd_fb.csv \
		-e NUM_PROC=30 \
		-v $$(pwd)/../../pipeline_test/input:/input:ro \
		-v $$(pwd)/../../pipeline_test/output:/output \
		-v $$(pwd)/../../pipeline_test/LDC2020E27:/kb \
		-v $$(pwd)/../../pipeline_test/temp:/aida/tmp \
		-v $$(pwd)/../../pipeline_test/res:/aida/res \
		-v $$(pwd):/aida/ta2-pipeline \
		--entrypoint /bin/bash gaiaaida/ta2