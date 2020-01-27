# Documentation:

This is one of 3 repositories for Matěj Račinský's master thesis, available [here](https://dspace.cvut.cz/handle/10467/76430).

There are 3 repositories:
- GTAVisionExport
    - the actual GTA mod which modifies the game
- GTAVisionExport_Server
    - repo with some scripts to control the data gathering and preview gathered images
- GTAVisionExport_postprocessing
    - whole postprocessing, most of interesting things is here

## Starting with the generate_bboxes postprocessing

Bunch of scripts for some simple post-processing of [GTAVisionExport](https://github.com/umautobots/GTAVisionExport) repo.

It is another kind of post-processing than is mentioned in their paper. For their original post-processing, 
see [this repo](https://github.com/umautobots/gta-postprocessing).

You need to connect to database where managed plugin stores data.
Copy file `gta-postprocessing.ini.example` to `gta-postprocessing.ini` and there fill the database credentials. 

### Using the original postprocessing for creating new stencil data
Script is the `generate_bboxes.py`. 
It does post-processing in batches, processes all images per run.
Run it like this:
```
python /datagrid/personal/racinmat/GTAVisionExport_postprocessing/generate_bboxes.py --pixel_path /datagrid/personal/racinmat/GTA-sample-data-annotated --run 60 --dataroot /datagrid/personal/racinmat/GTA-sample-data
```
The run parameter is run_id of snapshots.
This can not be run on Windows, because it uses libtiff library, which works only on linux. 
It stores calculated stencils as zipped numpy array, for loading, see `show_driving_in_the_matrix_stencil.ipynb`.


Note: pausing game has significant impact on timeofday flow, it looks like during classic 4camera data gathering, timeofday changing is 10 times slower.



## Notes about offroad:

Během offroadu se využívá automatické řízení z modu VAutodrive, který používá nativní pathfinding ve hře. 
Ten funguje pro offroadová vozidla i na offroadu, pro silniční vozidla  funguje pouze na silnicích.
I offroad auta mají tendenci jezdit spíše po silnicích, pokud jsou po cestě.

During offroad, automtic driving from VAutodrive is used. This uses the native ingame pathfinding.
That pathfinding works for all vehicles for onroad terraing, but for offroad terrain, it works sufficiently only for offroad vehicles.

Automatic driving has 2 modes:
- wandering around - car drives randomly on roads
- driving to a target - car heads to the goal using the pathfinding. The target is set as waypoint 
(there is method for in in the GTA API, you only set X, Y coordinates of the target). 
So you need to set a target manually (of by script) if you want to do the offroad driving.
When the start and target are far from roads, path is usually found fully offroad, but when it can, 
it prefers driving on roads while heading to the target.
When setting start and target randomly, the vehicle usually drives long distances on road. 

Offroad areas are defined in the `GTAVisionExport/managed/GTAVisionExport/OffroadPlanning.cs` from which starts and targets are sampled.
Actually, most of logic for offroad driving is in this script. The actual data gathering is shared with the onroad, and is in the 
`GTAVisionExport/managed/GTAVisionExport/VisionExport.cs` as usual.

To set car on ground for some selected start position, I need the X, Y and Z coordinates.
During the start and target sampling, only X and Y coordinates are selected, Z is implicit, it's the ground height on [X, Y].
To obtain Z ground coordinate for [X, Y], `World.GetGroundHeight` method is used, which uses raycasting from above in down direction 
to determine the ground height. 
But that works only when all textures are loaded in the area, and textures are loading only around the active camera 
(thus around the player character).
This is the current, working solution:
After selecting X, Y, player character is teleported in the air, high above ground, with [X, Y] coordinates few meters from the [X, Y] start.
(when doing raycasting, if character is in the ray trajectory, it is hit by the ray, so I must do raycasting in some place without character).
After teleporting, player character starts falling, and meanwhile, textures are loading. 
During the fall, I repeatedly perform the raycasting until I obtain the correct height (Z coordinate). 
Then I set car and its driver position to that position.
So far it worked every time and I obtained height before the character fell, but texture loading time differs, 
so I can't wait for some constant time before checking the height.
Alternate solution would be set car with driver high above ground, set them to invincible and let them fall.
But then car will turn on roof, drive away, deform, and so on...and it's more prone to error than current approach.

Settings up parameters of cameras used for offroad is in `GTAVisionExport/managed/GTAVisionExport/VisionExport.cs` in 
the initializeCameras method, when I uncomment cameras I want to use and comment out all remaining cameras 
(now I use it for switching between offroad and onroad cameras).

## Data generating process 

I gather data by driving in GTA V. RGB image, depth and stencil are stored as `.tiff` files in `D:\GTAV_extraction_output`, 
the rest is in PostgreSQL database in docker, it's docker-compose file is `GTAVisionExport\managed\docker-compose.yml`. 
The DB structure is in  `GTAVisionExport/db_schema.sql`.

In the DB, only tables snapshots and detections are used the most.
Each driving sequence in the GTA has its autoincremental identifier, run_id.
For all rows with same run_id belong to the same driving sequence, and usually are exported together.

For each image, I have stored all entities nearby. But their visibility in the image si note resolved.
Ëntities behind the camera, behind other objects, entities not shown in the image are saved for each snapshot, 
so for annotation and showing visible entities, filtering must be done. 
Filtering is done in the gta_math.py in function is_entity_in_image.
There are several criteria: 
If the entity is behind the camera in camera view coordinates, it is not in the image. 
If the bounding box of the entity is behind out of the [-1, 1] range in the NDC coordinates, it is not in the image. 
Entity is occluded and thus not in the image if ratio of car stencil pixels inside its 2d bounding box is less than threshold.
Then, all car stencil pixels are taken and their depth is compared with the car's 3d bounding box.
If the ratio of these depth pixels belonging into the bounding box is too low, car is occluded by other car and thus not in the image. 

The exported files (output of the prepare_generic_dataset) used for processing later are now mostly stored in the `D:\output-datasets`.
Original files are not deleted, but kept here. Because sometimes, same image sequence is generated multiple times, in different formats.
 

## Scripts list:

Most of runnable code is in jupyter notebooks, more used functions are in python files.

#### Python scripts

- visualization.py 
    - connection to database, things to visualize bounding boxes, lots of SQL commands to retrieve data frm DB
- gta_math.py
    - math around the GTA, transitions between coordinate systems, bounding boxes calculation, entities visibility...
    - few functions have unit tests 
- toyota.py
    - transfer of data into toyota format. They contain only rgb image, cars annotations and camera calibration data.
- kitti.py
    - transfer of rgb, depth and stencil images into the FOV of KITTI dataset cameras 
    - contains calculation of projection matrix for new FOV
- voxelmaps.py
    - transfer of scenes into voxelmaps (scene is composed from multiple cameras lookng at the same space)
- convert_generic_to_toyota_structure.py
    - utils script which takes the toyota data (which are all in one directory) and splits them into multiple directories, 
    camera per directory, images and entities annotations have their own directories

#### Jupyter notebooks
    
Most batch jobs are in jupyter notebooks. 
To use multithreading, joblib Parallel is used, and configuration is via global variables.

- prepare_generic_dataset.ipynb
    - to generate data into the more usable format, and dump data from DB into text files  
    - usually rgb is in jpg, stencil in png, depth in png or tiff, and db data are in json
    - only first few cells are being run frequently
    - to generate new sequence, I 
        - set the out_directory in the 2. cell
        - set the mode in the 2. cell
            - there are 3 modes, onroad, offroad, toyota, they group carious settings
        - set the run_id in 3. cell
        - run cells 1 - 5
    - copying into the linux datagrid is then done either by jupyter notebook or in windows explorer
    - settings variables
        - include_entities
            if true, entities (cars, pedestrians...) are included in the json
            false for offroad where only terrain matters
        - directory_per_camera
            if true, data from every camera are in its own directory
            used for offroad 
        - depth_in_tiff
            if true, depth is directly in tiff in NDC, not transformed into integers
            if false, it is stored in png as 16bit integers (0-1 range is mapped into the integer range)
            thus, some rounding errors occur
            export_depth - if false, depth is not copied
            export_stencil - if false, stencil is not copied
        - toyota_data_format
            if true, json data are transformed into the toyota format and json is not stored, only the resulting toyota file is stored. 
            If true, the calibration matrix in toyota format is stored too 

- dataset-images-to-videos.ipynb
    - creates videos from jpg images obtained by prepare_generic_dataset.ipynb
    - useful for overview of dataset, done per run (as most of things)
    - generates video per camera (almost all setups are multi-camera so far)
    - to generate new video, I
        - set the in_directory and out_directory in 2. cell
        - set the run_id
        - run cells 1 - 8
        
    - cells 9 and 10 are for generating video with bounding boxes, this takes lots of time because bounding box calculations take time
- annotation-examples.ipynb
    - there are examples of working with GTA coordinate systems, matrices and entities data
- sandbox.ipynb
    - there is lots of useful visualizations, displaying images...
- gta-dataset-readme.ipynb
    - very detailed explanation of GTA data, with description of paramers, description of coordinate systems and matrices...
    - very good, read it
    - example of code for manipulating data
- inspecting-broken-annotations.ipynb
    - showing images from datasets, with bounding boxes, good for inspecting bounding boxes
    
#### Other files
- gta-postprocesing.ini is the configuration file for connecting to the database

Other files, not described here, were not used for a long time and thus are not so important.
