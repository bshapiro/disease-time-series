#Preprocessing Tool

This is a preprocessing tool. You can give it raw data and have it perform simple cleaning operations: scaling/whitening, logtransforming, filtering out samples/features, regressing out covariates and principal components. All options for preprocessing can be specified via command-line (see below), or you can import the class directly for use in your own python script.

##ORDER OF OPERATIONS

1. Load the file, create preprocessing object
2. Log transform data if specified, smoothing by value of log_smoothing (default 0) 
3. Perform any filtering specified
4. Scale if specified
5. Data cleaning on pcs and values specified in regress_rows/regress_cols
6. Save output

##OPTIONS


###General:

    '--in_directory' specifies the directory to look for all files, use it if all your files are in one place
    '--out_directory' specifies the output directory to save all output
    '--savename' is the name of the preprocessed data ouput file
    '--saveformat' is the the format to save output file: 'pickle' or 'txt' which is tab separated. 
    '-d' or '--dataset' flag is the filepath/filename of the data matrix file
    '--filetype' is the filetype of -d, 'pickle' 'csv' 'tsv'
    '--has_row_labels', '--has_col_labels' boolean flags to interperet first column/row as labels
    '--transpose_data' whether or not to transpose matrix. NOTE: as convention we take rows to be samples and columns to be features


###Log Transforming:
    '-l' or '--log_transform' flag log transforms the data
    '--smoothing' specifies smoothing value to add elementwise to matrix before logtransformation (default=0)

###Scaling/Whitening:
    '-s' or '--scale_data' scales the data (by default to mean 0 and unit variance)
    '--scale_axis' specifies axis to scale on 0 for rows, 1 for columns (default 0)
    '--center_off' will scale data to unit variance without centering
    '--unit_std_off' will center data without scaling

###Filtering:
    '-f' or '--filter_data' allows you to filter your data. To perform multiple filterings use the '-f' flag as many times as you'd like with corresponding arguments.It requires 4 arguments:
        - the filename of a pickle of an array with the values that you want to filter on
        - one of our predefined operations. right now we have '==', '>', '>=', '<', '<=' but you can add easily by adding a new operation label and corresponding lambda in make_operation method
        - a value to threshold on
        - the axis to perform the filtering on. The dimension of the filter data should agree with the dimension of the data on this axis. 

###Regressing Out Covariates:
    '-r' or '--regress_out' flag allows you to regress out covariates. You will want the dimension of your values matrix and the axis you are regressing on to agree. DO NOT USE FLATTENED ARRAYS. eg. if we have an n x m matrix and want to regress on axis 0 use an n x k values matrix where k is the number of covariates we are regressing out. The flag requires 2 arguments:
        - the filename of a pickle of an array/matrix you want to regress your data against
        - the axis to do the regression on

    
###Regressing Out Principal Components:
    '-p' or '--principal_component' flag allows you to regress out a principal component by index. The flag requires one argument:
        - the index of the principal component to be regressed out



##EXAMPLES


```
python preprocessing.py --in_directory dir -d datafile --filetype tsv --has_row_labels --has_col_labels \
 -s -f filtervals1 '==' 0 0 -f filtervals2 '<=' 3 1 -p 0 -p 1 -p 2  --out_directory outdir --savetype txt
```

What it does:
- Will use tab separated textfile datafile in directory dir,
- Interperet datafile to have row and column labels,
- Scale and center the data,
- Filter samples(rows, axis 0) to values where filtervals1 == 0,
- Filter features(columns, axis 1) to values where filtervals2 <= 3,
- Regress out the first three principal components,
- Save output file as a tab separated text file to directory outdir

More examples to come!

##FUTURE IMPROVEMENTS:
- allow  covariates and filter data to be passsed in as .txt files
- allow results to be saved in MATLAB or R format
- add more examples to this README