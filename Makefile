.PHONY: docs 
docs:
	rm -rf docs/
	pdoc --html --output-dir docs --config show_source_code=False --config latex_math=True --force segmentation
	mv docs/segmentation/* docs
	rmdir docs/segmentation