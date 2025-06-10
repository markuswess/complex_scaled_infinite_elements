from IPython.nbconvert.preprocessors import *

class HyperlinkPreprocessor(Preprocessor):
    """docstring for HyperlinkPreprocessor"""

    def preprocess_cell(self, cell, resources, index):
        """
        Replaces *.ipynb hyperlinks with *.html.
        """
        
        if 'source' in cell and cell.cell_type == "markdown":
            cell.source = cell.source.replace('.ipynb','.html')
        return cell, resources
