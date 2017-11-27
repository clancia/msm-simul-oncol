c = get_config()
c.Exporter.preprocessors = [ 'bibpreprocessor.BibTexPreprocessor', 'pymdpreprocessor.PyMarkdownPreprocessor' ]
c.Exporter.template_file = 'thesis_template.tplx'