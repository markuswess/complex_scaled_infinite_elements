c = get_config()

#Export all the notebooks in the current directory to the sphinx_howto format.
c.FilesWriter.build_directory = 'html'
c.NbConvertApp.notebooks = ['*.ipynb']
c.Exporter.preprocessors = ['hyperlink_preprocessor.HyperlinkPreprocessor']
