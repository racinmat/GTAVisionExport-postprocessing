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


# Documentation:

There are 3 repositories:
- GTAVisionExport
    - the actual GTA mod which modifies the game
- GTAVisionExport_Server
    - repo with some scripts to control the data gathering and preview gathered images
- GTAVisionExport_postprocessing
    - whole postprocessing, most of interesting things is here


cam_front, cam_left, cam_rear, car_right
v nich složýky imgs, bboxes, a přímo ve složce kamery 
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
But that works only when all textures are loaded in the area.
Když vyberu start a chci auto teleportovat/posadit někam na zem v offroadu, musím znát X, Y, Z. 
Při vybírání startu a cíle sampluji jenom X, Y. 
Ta ale funguje jen, pokud jsou načtené všechny textury v okolí. Ty se načítají pouze v okolí hráče.
Momentálně funkční řešení:
po vybrání X, Y startu je postava hráče teleportována do vzduchu pár metrů od daných souřadnic 
(když dělám raycasting na pozici, kde je postava, paprsek se zastaví o ni a dozvím se tedy výšku postavy, ne země).
Během padání periodicky raycastuji, dokud nedostanu nenulovou hodnotu (nulová hodnota znamená neúspěšný raycasting). 
Na získanou Z souřadnici teleportuji postavu hráče v autě. 
Zatím to vždycky fungovalo a textury se načetly dříve než postava dopadla na zem.
Ale doba načtení se různí, takže nejde čekat pevný čas.

Alternativa by byla nastavit auto s řidičem na pozici vysoko nad zemí a nechat spadnout. Při nastavení invincible postavě,
 i autu přežijí pád. Ale auto se potluče, překotí, apod., a obecně dělá tento přístup mnohem větší neplechu než když se 
 auto posadí na zem.

Natavování kamer, které se používají při offroadu je GTAVisionExport/VisionExport.cs v metodě initializeCameras, 
nechám odkomentované jenom kamery, které chci používat, zbytek zakomentuji (např. když přepínám mezi sběrem pro onroad a offroad).

## Process generování dat 

Najezdím data v GTA. Ta jsou uložená v `D:\GTAV_extraction_output` jako tiffy, rgb, hloubka i stencil zvlášť. 
Zbytek dat je v PostgreSQL databázi, která je v dockeru. Struktura databáze je  

## Seznam skriptů:

Většina věcí je v jupyter noteboocích, sdílené funkce v python souborech.

#### python skripty

- visualization.py 
    - věci k cizualizaci aut, bounding boxů apod., a SQL prákazy na získávání dat, připojení k databázi
- gta_math.py
    - matematika okolo gta, převody souřadnic, počítání bounding boxů, viditelností entit apod. Pár funkcí má unit testy.
- toyota.py
    - převod dat do toyota formátu. Ty obsahují pouze rgb obrázek, anotace aut a kalibrační matici
- kitti.py
    - převod rgb, depth, stencilu na FOV odpovídající KITTI datasetu. Obsahuje i přepočet projekční matice.
- voxelmaps.py
    - převod scén do voxelových map (scéna je složena ze snímků více kamer, které se dívají na stejnou oblast z různých úhlů)

#### jupyter notebooky
    
- prepare_generic_dataset.ipynb  
    - obsahuje p
