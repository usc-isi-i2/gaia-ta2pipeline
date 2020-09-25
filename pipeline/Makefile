docker-build:
	docker build -t gaia-ta2-test .

docker-clean:
	docker rmi -f $$(docker images -f "dangling=true" -q)

docker-run:
	docker run --rm -it \
       	-e INPUT=/input \
       	-e OUTPUT=/output \
       	-e REPO_KB=/kb \
       	-e RUN_NAME=m36 \
       	-e TEMP=/aida/tmp \
       	-e NAMESPACE=/aida/res/aida-namespaces.tsv \
       	-e WD_FB_MAPPING=/aida/res/df_wd_fb.csv \
       	-e NUM_PROC=6 \
       	-v $$(pwd)/../pipeline_test/input:/input:ro \
       	-v $$(pwd)/../pipeline_test/output:/output \
       	-v $$(pwd)/../pipeline_test/LDC2020E27:/kb \
       	-v $$(pwd)/../pipeline_test/temp:/aida/tmp \
       	-v $$(pwd)/../pipeline_test/res:/aida/res \
       	-v $$(pwd):/aida/ta2-pipeline \
       	--entrypoint /bin/bash gaia-ta2-test
        
