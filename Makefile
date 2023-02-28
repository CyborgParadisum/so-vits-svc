ifneq ("$(wildcard .env)","")
	include .env
	export
endif

update:
	rsync -av  \
		--exclude-from .gitignore \
		--exclude .git \
		--exclude logs \
		--exclude configs \
		--exclude raw \
		--exclude dataset \
		--exclude dataset_raw \
		--exclude filelists \
		--exclude voice_source \
		--rsh="ssh -o StrictHostKeyChecking=no -p $(PORT)" \
		./* \
		$(DEPLOY_HOST):$(DEPLOY_PATH)
