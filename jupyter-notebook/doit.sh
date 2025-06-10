rm html/*
cp dissertation_wess.pdf html/dissertation_wess.pdf

jupyter nbconvert --config ipython_nbconvert_config.py

zip html/fq_dp_cs_ie.zip *.py *.ipynb html/*.html dhankel_1_zeros.out *.pdf -x ipython_nbconvert_config.py hyperlink_preprocessor.py
