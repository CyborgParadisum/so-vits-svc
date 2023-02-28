ifneq ("$(wildcard .env)","")
	include .env
	export
endif
define wget_if_not_exist
	@if [ ! -f $(1) ]; then \
		mkdir -p $(dir $(1)); \
		wget -O $(1) $(2); \
	fi
endef
download-model-hubert:
	$(call wget_if_not_exist, hubert/hubert-soft-0d54a1f4.pt ,\
			https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)

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
