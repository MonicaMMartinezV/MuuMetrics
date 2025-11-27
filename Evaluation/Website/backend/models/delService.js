const fs = require("fs");
const Papa = require("papaparse");
const path = require("path");

// =====================================================
// 1) Load individual cow CSV (equivalent to combinedDfVacas but per cow)
// =====================================================
function loadIndividualCowCsv(csvPath, cowId) {
    const fileContent = fs.readFileSync(csvPath, "utf8");

    const parsed = Papa.parse(fileContent, {
        dynamicTyping: false,
        skipEmptyLines: true
    });

    const rows = parsed.data;
    if (!rows.length) throw new Error("Empty individual CSV");

    // Find header row containing "Hora de inicio"
    const headerIndex = rows.findIndex(row =>
        row.some(col => String(col).includes("Hora de inicio"))
    );

    if (headerIndex === -1) {
        throw new Error("Header row ('Hora de inicio') not found");
    }

    // Data starts AFTER header row
    const header = rows[headerIndex];
    const dataRows = rows.slice(headerIndex + 1);

    // Build objects with headers
    const objects = dataRows.map(row => {
        const obj = {};
        header.forEach((col, i) => {
            obj[col] = row[i];
        });
        return obj;
    });

    // Remove unnamed columns
    const cleaned = objects.map(obj => {
        const o = {};
        for (const key of Object.keys(obj)) {
            if (!key.startsWith("Unnamed")) o[key] = obj[key];
        }
        return o;
    });

    // Insert vacaId
    cleaned.forEach(o => (o.vacaId = cowId));

    // Keep only the needed columns
    const result = cleaned.map(o => ({
        vacaId: o.vacaId,
        "Hora de inicio": o["Hora de inicio"],
        "Duración (mm:ss)": o["Duración (mm:ss)"]
    }));

    return result;
}

// =====================================================
// 2) Convert columns to Date + compute Hora de fin
// =====================================================
function processIndividualDf(df) {
    return df.map(row => {
        let start = row["Hora de inicio"];

        if (!start) return null;

        // Cleanup (equivalent to Python)
        start = String(start)
            .trim()
            .replace(/\s+/g, " ")
            .replace(/a\.? ?m\.?/gi, "AM")
            .replace(/p\.? ?m\.?/gi, "PM")
            .replace(/\./g, "");

        const startDate = new Date(start);
        if (isNaN(startDate.getTime())) return null;

        // CRITICAL FIX: Ensure both parts of duration are numbers before calculating
        const durationStr = row["Duración (mm:ss)"] || "0:00";
        const [mm, ss] = durationStr.split(":").map(s => Number(s) || 0);

        const durationMs = (mm * 60 + ss) * 1000;

        const endDate = new Date(startDate.getTime() + durationMs);
        
        // Final check: if duration was bad, endDate might be Invalid Date (NaN), 
        // which will be filtered by getIdFromIndividual's date comparison, but 
        // better to check here too.
        if (isNaN(endDate.getTime())) return null;


        return {
            vacaId: row.vacaId,
            start: startDate,
            end: endDate
        };
    }).filter(Boolean);
}

// =====================================================
// 3) Load patadas.csv
// =====================================================
function loadPatadasCsv(csvPath) {
    const fileContent = fs.readFileSync(csvPath, "utf8");

    const parsed = Papa.parse(fileContent, {
        dynamicTyping: false,
        header: true,
        skipEmptyLines: true
    });

    return parsed.data.map(row => {
        let timeStr = String(row["Hora Inicio Ordeño"]).trim();

        // Normalize AM/PM formatting and spacing
        timeStr = timeStr
            .replace(/a\.? ?m\.?/gi, "AM")
            .replace(/p\.? ?m\.?/gi, "PM")
            .replace(/\s+/g, " ");

        const numero = Number(row["Número del animal"]);
        const delValue = Number(row["DEL"]);

        // --- NEW ROBUST DATE PARSING FOR DD/MM/YYYY HH:MM AM/PM ---
        const parts = timeStr.split(' '); // e.g., ["18/07/2025", "07:53", "AM"]
        let horaDate = new Date('Invalid'); // Initialize as Invalid Date

        if (parts.length === 3) {
            const [date, time, meridiem] = parts;
            const dateParts = date.split('/').map(p => parseInt(p, 10)); // [18, 7, 2025]
            const timeParts = time.split(':').map(p => parseInt(p, 10)); // [7, 53]

            if (dateParts.length === 3 && timeParts.length === 2) {
                let [day, month, year] = dateParts;
                let [hour, minute] = timeParts;

                // Convert 12h to 24h format
                if (meridiem.toUpperCase() === 'PM' && hour < 12) {
                    hour += 12;
                } else if (meridiem.toUpperCase() === 'AM' && hour === 12) { // 12:xx AM (midnight) becomes 00:xx
                    hour = 0;
                }
                
                // Construct Date: YYYY, MM (0-indexed), DD, HH, MM, SS
                horaDate = new Date(year, month - 1, day, hour, minute, 0); 
            }
        }
        // --- END NEW ROBUST DATE PARSING ---
        
        // FIX 1: Check if DEL is valid
        if (isNaN(delValue)) {
            console.warn(`Skipping row: DEL is NaN for animal number ${numero}. Row data: ${JSON.stringify(row)}`);
            return null; // Skip this row
        }
        
        // FIX 2: Check if Hora Inicio Ordeño is a valid date (now using the robustly parsed date)
        if (isNaN(horaDate.getTime())) {
            console.warn(`Skipping row: Invalid date for "Hora Inicio Ordeño" (${timeStr}) for animal number ${numero}.`);
            return null; // Skip this row, as it will cause NaN in subtraction
        }

        return {
            numero: numero,
            DEL: delValue,
            hora: horaDate
        };
    }).filter(Boolean); // Filter out any nulls returned due to bad data
}

// =====================================================
// 4) Parse date from image filename
// =====================================================
function getImageDate(imagePath) {
    const name = path.basename(imagePath);
    
    // The date/time string is the second element after splitting by '_' (index 1)
    const dateTimeWithExt = name.split("_")[1]; 

    // Guard clause
    if (!dateTimeWithExt) {
        throw new Error(`Invalid image filename format: Expected "ID_YYYY-MM-DD-HH-MM-SS...", got "${name}"`);
    }

    // Remove file extension from the date/time string before parsing.
    // E.g., "2025-08-08-07-22-11.jpg" -> "2025-08-08-07-22-11"
    const dateTimePrefix = path.parse(dateTimeWithExt).name;
    
    // FIX: Convert YYYY-MM-DD-HH-MM-SS to YYYY-MM-DDTHH:MM:SS for reliable Date parsing
    const parts = dateTimePrefix.split('-'); 
    let dateString;
    
    if (parts.length === 6) {
        // Construct ISO-like string
        dateString = `${parts[0]}-${parts[1]}-${parts[2]}T${parts[3]}:${parts[4]}:${parts[5]}`;
    } else {
        // Fallback for unexpected formats, though this should be an error case
        dateString = dateTimePrefix.replace(/-/g, "/"); 
    }
    
    const dt = new Date(dateString);
    
    if (isNaN(dt.getTime())) {
        throw new Error(`Invalid date format in image filename: Failed to parse date string "${dateString}" from original component "${dateTimeWithExt}".`);
    }
    return dt;
}

// =====================================================
// 5) Get ID based on image time (equivalent to getIdImag)
// =====================================================
function getIdFromIndividual(individualDf, imgDt) {
    // Ensure df is sorted by end time
    const sorted = [...individualDf].sort((a, b) => a.end.getTime() - b.end.getTime()); 
    // Find all records that ended *before* the image timestamp
    const before = sorted.filter(row => row.end < imgDt);

    if (before.length === 0) return null;

    // The last entry in 'before' is the most recent activity record before the image
    return Number(before[before.length - 1].vacaId);
}

// =====================================================
// 6) Compute DEL (equivalent to getDEL)
// =====================================================
function computeDEL(patadasDf, ID, imgDt) {
    const row = patadasDf.find(r => r.numero === ID);
    if (!row) throw new Error(`Cow ID ${ID} not found in patadas (check loadPatadasCsv warnings)`);

    const baseDEL = row.DEL;
    const baseTime = row.hora;
    
    // FIX 3: Check baseDEL right away. If it's NaN, the result will be NaN.
    if (isNaN(baseDEL)) {
        // This case should be caught by loadPatadasCsv, but checking again for safety
        throw new Error(`Base DEL value is NaN for cow ID ${ID} in patadas data.`);
    }

    // This subtraction yields milliseconds. If either date is invalid, this is NaN.
    const timeDiffMs = imgDt - baseTime; 
    if (isNaN(timeDiffMs)) {
        throw new Error(`Date subtraction resulted in NaN. Check if image date (${imgDt.toString()}) or base time (${baseTime.toString()}) are Invalid Dates.`);
    }

    // Conversion factor for milliseconds to days
    const MS_PER_DAY = 1000 * 60 * 60 * 24; 
    
    // Calculate difference in days
    const diffDays = Math.floor(timeDiffMs / MS_PER_DAY);

    return baseDEL + diffDays;
}

// =====================================================
// 7) Main function
// =====================================================
async function computeDELFromFiles(cowId, individualCsvPath, patadasCsvPath, imagePath) {
    const rawDf = loadIndividualCowCsv(individualCsvPath, cowId);
    const df = processIndividualDf(rawDf);

    const patadasDf = loadPatadasCsv(patadasCsvPath); 

    const imgDt = getImageDate(imagePath);

    const ID = getIdFromIndividual(df, imgDt);
    if (!ID) throw new Error("Could not determine cow ID for image based on individual CSV records.");

    const DEL = computeDEL(patadasDf, ID, imgDt);

    return { ID, DEL };
}

exports.computeDELFromFiles = computeDELFromFiles;