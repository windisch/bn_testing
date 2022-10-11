# Configuration file for the Sphinx documentation builder.

with open('../../VERSION') as f:
    version = f.read()


release = version
project = 'bn_testing'
copyright = '2022'
author = 'Tobias Windisch'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}
intersphinx_disabled_domains = ['std']
autosectionlabel_prefix_document = True

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'


try:
    import os
    import shutil
    from sphinx.ext import apidoc

    curdir = os.path.dirname(__file__)
    dir_api = os.path.join(curdir, 'generated')
    dir_module = os.path.join(curdir, '../../bn_testing/')

    shutil.rmtree(dir_api, ignore_errors=True)

    cmd = f"-f -o {dir_api} {dir_module}"
    apidoc.main(cmd.split(' '))


except Exception as e:
    print("Failed to build api-docs: {}".format(e))
