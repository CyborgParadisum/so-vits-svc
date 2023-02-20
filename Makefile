ifneq ("$(wildcard .env)","")
	include .env
	export
endif

define wget_if_not_exist
	@if [ ! -f $(1) ]; then \
		wget -O $(1) $(2); \
	fi
endef

download-model:
	$(call wget_if_not_exist, hubert/hubert-soft-0d54a1f4.pt, https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)
	mkdir -p models/
	$(call wget_if_not_exist, models/mikisayaka-G_50000.pth, https://huggingface.co/pengyifu/so-vits-svc/resolve/main/G_50000.pth)
	$(call wget_if_not_exist, models/mikisayaka-config.json, https://huggingface.co/pengyifu/so-vits-svc/resolve/main/config.json)
	@#wget -P hubert/ https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt
	@#wget -O models/mikisayaka-G_50000.pth https://huggingface.co/pengyifu/so-vits-svc/resolve/main/G_50000.pth
	@#wget -O models/models/mikisayaka-G_50000.pth https://huggingface.co/pengyifu/so-vits-svc/resolve/main/config.json

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
