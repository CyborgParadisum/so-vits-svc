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

modules_dir = _models
download-model:
	$(call wget_if_not_exist, hubert/hubert-soft-0d54a1f4.pt, https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
	mkdir -p $(modules_dir)
	$(call wget_if_not_exist, $(modules_dir)/mikisayaka/mikisayaka-G_50000-infer.pth ,\
			https://huggingface.co/SuCicada/SuTTS/resolve/main/mikisayaka/mikisayaka-G_50000-infer.pth)
	$(call wget_if_not_exist, $(modules_dir)/mikisayaka/mikisayaka-config.json ,\
			https://huggingface.co/SuCicada/SuTTS/resolve/main/mikisayaka/mikisayaka-config.json)

	$(call wget_if_not_exist, $(modules_dir)/sakurakyouko/sakurakyouko-G_100000-infer.pth ,\
			https://huggingface.co/SuCicada/SuTTS/resolve/main/sakurakyouko/sakurakyouko-G_100000-infer.pth)
	$(call wget_if_not_exist, $(modules_dir)/sakurakyouko/sakurakyouko-config.json ,\
			https://huggingface.co/SuCicada/SuTTS/resolve/main/sakurakyouko/sakurakyouko-config.json)

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
