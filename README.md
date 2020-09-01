# M-LARGE
![GitHub last commit](https://img.shields.io/github/last-commit/jiunting/MLARGE?style=plastic)  
### Machine-learning Assessed Rapid Geodetic Earthquake magnitude   
A deep-learning based mega-earthquake magnitude predictor  
* [1. Installation](#1-Installation)
  * [1.1 Download M-LARGE](#cd-to-the-place-where-you-want-to-put-the-source-code)
  * [1.2 Add environment variable](#Add-M-LARGE-to-PYTHONPATH)
* [2. Generate rupture and seismic waveforms data](#2-Generating-rupture-and-seismic-waveforms-data)  
* [3. Run M-LARGE training](#3-Run-M-LARGE-training)
  * [3.1 Check data structure](#Before-running-the-M-LARGE-make-sure-you-have-data-following-Mudpys-structure)
  * [3.2 Set the paths in control file](#Change-all-the-paths-in-the-controlpy-file)
  * [3.3 Variables in control file](#Below-shows-the-variables-and-their-corresponding-meaning)

![][Exp_fig2] 

****
## 1. Installation
#### cd to the place where you want to put the source code  
```console
cd Your_Local_Path  
git clone https://github.com/jiunting/MLARGE.git
```
#### Add M-LARGE to PYTHONPATH

> Go to your environval variable file (.base_profile or .bashrc)  
```console
vi ~/.bashrc  
```
> or  
```console
vi ~/.bash_profile      
```
> and add the following line in the file

```bash
#set MLARGE
export PYTHONPATH=$PYTHONPATH:YOUR_PATH_MARGE/MLARGE/src
```    

****
## 2. Generating rupture and seismic waveforms data
#### Earthquake scenario and synthetic seismic waves are based on the below methods  
* Melgar, D., LeVeque, R. J., Dreger, D. S., & Allen, R. M. (2016). Kinematic rupture scenarios and synthetic displacement data: An example application to the Cascadia subduction zone. Journal of Geophysical Research: Solid Earth, 121(9), 6658-6674.  

* Melgar, D., Crowell, B. W., Melbourne, T. I., Szeliga, W., Santillan, M., & Scrivner, C. (2020). Noise characteristics of operational real‐time high‐rate GNSS positions in a large aperture network. Journal of Geophysical Research: Solid Earth, 125(7), e2019JB019197.

> You can make your own rupture scenarios and waveforms. The Mudpy source code can be downloaded [HERE][Mudpy]  

> Or download the example data directly from [HERE][Link_data]

****
## 3. Run M-LARGE training  
#### Before running the M-LARGE, make sure you have data following Mudpy's structure  
```
-home_directory
    -project_name
        -output
            -ruptures
            -waveforms
```
#### For example, this is my data structure for rupture scenarios  

![][Exp_fig1] 

#### In this case:  
>*home_directory is the absolute path before the ```Chile```  (e.g. /Users/timlin/Test_MLARGE/)  
>*project_name is ```Chile```  
>*output, ruptures and waveforms should be exactly the name ```output; ruptures; waveforms```

#### Make a project directory
```console
mkdir Project_Name
cd Project_Name
cp YOUR_PATH_MARGE/example/control.py . #copy example file to your project directory
```
#### Change all the paths in the `control.py` file  
```python
home = PATH_To_Project_Name  #without the project name
project_name = Project_Name_For_Data  #for data, not the Project_Name above
run_name = Prepend_Name
```
#### Check the above path by testing in python
```python
>>print(home+project_name+'/output/ruptures/')  
>>print(home+project_name+'/output/waveforms/')  
```
> These should point you to the right directory to *.rupt/*.log and waveforms directory.   
> The rupt files are named in run_name.xxxxxx.rupt.  
#### Below shows the variables in the control file and their corresponding meaning
|Variable Name  |Meaning |
| :---------- | :-----------|
| save_data_from_FQ   |True/False- Write E/N/Z.npy data and STFs from the above waveforms/ruptures directories   |
| gen_list   |True/False- Generate data path for later training   |
| gen_EQinfo |True/False- Read above rupture files and generate rupture information file  |
| tcs_samples|Array of the time for ENZ and STF |
| outdir_X| Output directory name for E/N/Z.npy|
| outdir_y| Output directory name for STF|
| out_list| Prepended output name for E/N/Z/STF list file|
| out_EQinfo| Output name for rupture information|
| GFlist| Path of GFlist that used to generate rupture scenarios|
| Sta_ordering| Path of station ordering file (keep always the same order while in training/prediction)|

****

[Mudpy]:https://github.com/dmelgarm/MudPy "Multi-data source modeling and inversion toolkit"
[FK]:http://www.eas.slu.edu/People/LZhu/home.html "FK package from Dr. Zhu Lupei"
[Link_data]:https://zenodo.org/ "Data will be released soon..."
[Exp_fig1]:/image/Exp_datastructure.png "Mudpy data structure"
[Exp_fig2]:/image/Exp_architecture.png "MLARGE model structure"

