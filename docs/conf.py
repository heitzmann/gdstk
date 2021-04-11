#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2020 Lucas Heitzmann Gabrielli.
# This file is part of gdstk, distributed under the terms of the
# Boost Software License - Version 1.0.  See the accompanying
# LICENSE file or <http://www.boost.org/LICENSE_1_0.txt>

import sys
import pathlib
import distutils.command.build
from distutils.dist import Distribution

# Update path so that autodoc is able to find the module
empty_build = distutils.command.build.build(Distribution())
empty_build.initialize_options()
empty_build.finalize_options()
path = pathlib.Path("..") / empty_build.build_platlib
sys.path.insert(0, str(path.absolute()))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_inline_tabs",
]

autosummary_generate = True
autosummary_imported_members = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

exclude_patterns = ['Thumbs.db', '.DS_Store']

html_static_path = ['_static']

templates_path = ["_templates"]

pygments_style = "trac"

html_copy_source = False

html_show_sphinx = False

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    #'canonical_url': '',
    #'analytics_id': '',
    #'logo_only': False,
    "display_version": True,
    #'prev_next_buttons_location': 'bottom',
    #'style_external_links': False,
    #'vcs_pageview_mode': '',
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": -1,
    #'includehidden': True,
    #'titles_only': False
}
