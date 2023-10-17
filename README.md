### grand_psu_analysis
Repo for the analysis of GP13 and G@A data from the PSU team. 


## Structure of the repository

The directory grand_psu_lib contains the core of the codes. It contains two directories, utils and modules.
- Utils contains the utility scripts
- Modules contains the core of the library

The directory scripts contains the scripts used for specific analysis. It is those scripts that use the code contained in the grand_psu_lib directory.

The directory tools contains command-line scripts that can use to perform quick and repetitive analysis of GRAND files.


## Usage

The only file to run is tools/large_scale_diagnostic.py
The command is
```
large_scale_diagostic.py  --file_path FILE_PATH --plot_path PLOT_PATH --site SITE [--do_fourier_vs_time DO_FOURIER_VS_TIME] [--base BASE]
```

```
file_path``` is the path of ther root files to analyse. It is fed to glob in the code so that it ios possible to analyse several files of a same run.
Possible syntax are given below. In case of widl card or [], it is necessary to use quotes. 

```
--file_path "$PATH/data/auger/TD/td002022_f000[1-2].root"
--file_path $PATH/data/auger/TD/td002022_f0001.root
--file_path "$PATH/data/auger/TD/td002022_f000*.root"
```


