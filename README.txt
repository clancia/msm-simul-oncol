To run the thesis.ipynb file the following five files must be available

The first three come from:

http://blog.juliusschulz.de/blog/ultimate-ipython-notebook

under the Converting notebook to LaTeX/PDF section

1. bibpreprocessor.py  
2. jupyter_nbconvert_config.py
3. pymdpreprocessor.py

The last two are made by the user

4. thesis.bib
5. thesis.ipynb

To convert thesis.ipynb to pdf go to the folder where the file is stored and run the following from the command line 

jupyter nbconvert --to=latex thesis.ipynb
