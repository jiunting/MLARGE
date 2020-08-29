# M-LARGE

### Machine-learning Assessed Rapid Geodetic Earthquake magnitude   
An deep-learning based mega-earthquake magnitude predictor

****
## 1. Installation
> cd to the place where you want to put the source code  

    cd Your_Local_Path  
    git clone https://github.com/jiunting/MLARGE.git
    
> Add M-LARGE to PYTHONPATH

> Go to your environval variable file (.base_profile or .bashrc)  

    vi ~/.bashrc  
    
> or 

    vi ~/.bash_profile      
    
> and add the following line in the file

```bash
#set MLARGE
export PYTHONPATH=$PYTHONPATH:YOUR_PATH_MARGE/MLARGE/src
```    
****

## 2. Generating seismic waveforms data
> Earthquake scenario and synthetic seismic waves are based on the below methods  
* Melgar, D., LeVeque, R. J., Dreger, D. S., & Allen, R. M. (2016). Kinematic rupture scenarios and synthetic displacement data: An example application to the Cascadia subduction zone. Journal of Geophysical Research: Solid Earth, 121(9), 6658-6674.  

* Melgar, D., Crowell, B. W., Melbourne, T. I., Szeliga, W., Santillan, M., & Scrivner, C. (2020). Noise characteristics of operational real‐time high‐rate GNSS positions in a large aperture network. Journal of Geophysical Research: Solid Earth, 125(7), e2019JB019197.

> You can make your own rupture scenarios and waveforms. The Mudpy source code can be downloaded [HERE][Mudpy]  

> Or download the example data directly [HERE][Link_data]

## 3. Run M-LARGE training  
> Before running the M-LARGE, make sure you have data following Mudpy's structure  
```
-home_directory
    -project_name
        -output
            -ruptures
            -waveforms
```
> For example, this is my rupture scenarios  


> Make a project directory
```bash
mkdir Project_Name
cd Project_Name
cp YOUR_PATH_MARGE/example/control.py . #copy example file to your project directory
```
> Change all the paths in the `control.py` file


[Mudpy]:https://github.com/dmelgarm/MudPy "Multi-data source modeling and inversion toolkit"
[FK]:http://www.eas.slu.edu/People/LZhu/home.html "FK package from Dr. Zhu Lupei"
[Link_data]:https://zenodo.org/ "Data will be released soon..."
