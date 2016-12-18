To run this package, you will need the following installed:

- numpy
- scipy
- GPy
- matplotlib
- seaborn
- pandas

Most of these packages are already installed on gradx. The exceptions are seaborn and GPy; to install, run: 

pip install --local GPy
pip install --local seaborn

Because GPy depends on some of these packages, it may require that you reinstall them locally (which it should do for you.) 

To a run a demo, run the bash script. It will run the single view myeloma-paper initialized version of the ribosome data clustering, and then it will plot the corresponding heat map for the clustering. Likelihoods will print to stdout during the course of the run.

To run other configurations, you will need to change the config.py file (there are comments to explain what each parameter does). 

To generate the heat map for the configuration, you will need to run generate+heat+map.py, and the first command-line argument must be the directory containing the memberships.dump file for the run. For example:

python singleview.py
python generate+heat+map.py single/kmeans/avg/raw/clusters5/polya 

*changes config*

python twoview.py
python generate+heat+map.py two/myeloma_paper/avg/raw/clusters5/ribosome. 

