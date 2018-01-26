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