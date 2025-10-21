# Código realizado:

## calculoDELvacas.py

### Overview
The script *calculoDELvacas.py* computes the *Days in Milk (DEL)* closest to the moment a cow’s image was taken.  
It combines milking records from multiple cows (each stored in a separate .csv file) and matches them to the timestamps of images to identify which cow and milking session are closest in time.

---

### How It Works:

1. *Reads and combines CSV files*
   - The script scans all .csv files inside the directory defined by pathCows.
   - Each file represents one cow, and its filename is used as the cow’s ID.
   - It finds the correct header row, cleans up the data, and merges all files into one combined DataFrame.

2. *Cleans and converts date columns*
   - The milking start times (Hora de inicio) are standardized and converted to datetime format.
   - Any missing or invalid date entries are reported to the user.

3. *Processes image timestamps*
   - A dummy list of image filenames (e.g., 2025-06-01-21-47-55_cam4_cap3) is converted into datetime objects in case the images are converted to the correct format all at once.
   - The script checks for invalid or unrecognized image names and lists the valid ones.

4. *Reads “Patadas” dataset*
   - The file defined in pathPatadas is read.  
   - This file must contain the columns:
     - Número del animal
     - DEL
     - Hora Inicio Ordeño

5. *Finds the closest matching cow*
   - For a target image timestamp, the script finds which cow’s milking record (Hora de inicio) is closest in time.
   - It then retrieves that cow’s DEL value and prints:
     - The cow ID
     - The closest matching datetime
     - The calculated DEL value corresponding to the image date

6. *Calculates adjusted DEL*
   - The DEL value is adjusted based on the difference (in days) between the milking record and the image timestamp.
   - The output shows how many days in milk the cow had at the time the image was taken.

---

### Requirements:

The script uses the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- glob
- datetime
- os

You can install them with:

```bash
pip install numpy pandas matplotlib seaborn glob datetime os
