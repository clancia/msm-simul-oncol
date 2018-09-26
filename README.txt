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

rpy2

To use rpy2.ipython package you need to set the R_HOME environment variable to 
C:\Program Files\R\R-3.4.3 or equivalent to access the C:\Program Files\R\R-3.4.3\bin\x64\R.dll file

In addition, make sure to copy the C:\Program Files\R\R-3.4.3\bin\x64\Rlapack.dll file into both the C:\Program Files\R\R-3.4.3\library\stats\libs\x64 and C:\Program Files\R\R-3.4.3\library\Matrix\libs\x64 folders.

The following R packages must be loaded: survey, ipw

