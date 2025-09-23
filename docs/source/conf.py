# Configuration file for the Sphinx documentation builder
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Podcast LLM'
copyright = 'Evan Dempsey 2024'
author = 'Evan Dempsey'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] 
