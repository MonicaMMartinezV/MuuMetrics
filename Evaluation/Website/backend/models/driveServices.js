const { google } = require("googleapis");
const { computeDELFromFiles } = require("./delService");
const path = require("path");
const fs = require("fs");
const KEY_FILE = "credentials.json";
const folderInfo = require("../../folder_info.json");
const csv = require("csv-parser");
const { get } = require("http");

const FOLDER_ID = folderInfo.folder_id;
// Define the path for downloaded files
const DOWNLOAD_DIR = path.join(__dirname, "..", "utils", "downloads");

// Define the necessary Google API scopes
const SCOPE = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
];

// -------------------------------------------------------------
// DRIVE CLIENT & BASIC GET FILES
// -------------------------------------------------------------

/**
 * Creates and authenticates a Google Drive client.
 * @returns {Promise<google.drive.Drive>} The initialized Drive client.
 */
async function getDriveClient() {
    try {
        const auth = new google.auth.GoogleAuth({
            keyFile: KEY_FILE,
            scopes: SCOPE,
        });
        const client = await auth.getClient();
        return google.drive({ version: "v3", auth: client });
    } catch (error) {
        console.error("Error creating Drive client:", error.message);
        throw error;
    }
}

/**
 * Lists all files within the 'img' subfolder of the main FOLDER_ID.
 * @returns {Promise<Array<Object>>} List of file objects.
 */
async function getFiles() {
    const drive = await getDriveClient();
 // 1. Find the "img" folder inside your root folder
    console.log("Searching for 'img' folder inside root folder...");

    const folderSearch = await drive.files.list({
        q: `'${FOLDER_ID}' in parents and name='img' and mimeType='application/vnd.google-apps.folder' and trashed=false`,
        fields: "files(id, name)",
    });

    if (!folderSearch.data.files.length) {
        console.log("ERROR: No 'img' folder found.");
        return [];
    }

    const imgFolderId = folderSearch.data.files[0].id;

    console.log("Image folder found (ID: " + imgFolderId + ")");

    // 2. Get ALL images inside the IMG folder
    const imgSearch = await drive.files.list({
        q: `'${imgFolderId}' in parents and mimeType contains 'image/' and trashed=false`,
        fields: "files(id, name, mimeType)",
    });

    if (!imgSearch.data.files.length) {
        console.log("No images found inside img/");
        return [];
    }
    console.log(`Found ${imgSearch.data.files.length} image(s) inside img/ folder.`);

    return imgSearch.data.files;
}

async function getImageDates() {

    const files = await getFiles();
    console.log("Files:", files);

    if (!files.length) {
        throw new Error("No image files found inside 'img' folder.");
    }

    const results = [];

    for (const file of files) {
        try {
            const name = file.name;
            console.log("Processing:", name);

            // Extract ALL numeric blocks in order
            // Example: "2025-07-15-07-50-44_cam0_cap4.jpg"
            // → ["2025", "07", "15", "07", "50", "44"]
            const numbers = name.match(/\d+/g);

            if (!numbers || numbers.length < 6) {
                console.warn(`Skipping invalid filename (not enough numbers): ${name}`);
                continue;
            }

            const [year, month, day, hour, minute, seconds] = numbers;

            const dateString = `${year}-${month}-${day}T${hour}:${minute}:${seconds}`;
            const dt = new Date(dateString);

            if (isNaN(dt.getTime())) {
                console.warn(`Skipping invalid date in filename: ${name}`);
                continue;
            }

            results.push({
                name,
                date: dt,
            });

        } catch (err) {
            console.warn("Error processing file:", file, err);
            continue;
        }
    }
    console.log("Image dates extracted:", results);
    return results;
}

function pad(n) {
    return n.toString().padStart(2, "0");
}

function parsePatadaDate(dateStr) {

    if (!dateStr) return NaN;

    const [datePart, timePart, ampm1, ampm2] = dateStr.split(" ");
    // NOTE: "a. m." splits into ["a.", "m."]

    const ampmPart = (ampm1 + ampm2).toLowerCase(); // "a.m." or "p.m."

    const [day, month, year] = datePart.split("/").map(Number);
    let [hour, minute] = timePart.split(":").map(Number);

    const isPM = ampmPart.includes("p");

    // Convert to 24h format
    if (isPM && hour !== 12) hour += 12;
    if (!isPM && hour === 12) hour = 0;

    const isoString = `${pad(year)}-${pad(month)}-${pad(day)}T${pad(hour)}:${pad(minute)}:00`;
    console.log("ISO:", isoString);

    return new Date(isoString).getTime();
}





async function getIdImag() {
    const imgData = await getImageDates();

    if (!imgData || imgData.length === 0) {
        throw new Error("No valid image dates found.");
    }

    // Sort newest → oldest is optional; we will still iterate all
    //imgData.sort((a, b) => b.date - a.date);

    // Load Patadas CSV
    const { localPath } = await downloadPatadas();

    // Read CSV
    const cowData = await new Promise((resolve, reject) => {
        const rows = [];
        fs.createReadStream(localPath)
            .pipe(csv())
            .on("data", (row) => rows.push(row))
            .on("end", () => resolve(rows))
            .on("error", reject);
    });

    console.log("CSV rows loaded:", cowData.length);

    // Convert cow row times to timestamps for speed
    const formattedCowData = cowData.map((row) => ({
        id: row["Número del animal"],
        end: parsePatadaDate(row["Hora Inicio Ordeño"])
    }));

    // Build result array
    const results = [];

    for (const img of imgData) {
        const imgTime = img.date.getTime();

        let matchedId = null;

        // Iterate backward to find the last cow event before image
        for (let i = formattedCowData.length - 1; i >= 0; i--) {
            if (formattedCowData[i].end < imgTime) {
                matchedId = formattedCowData[i].id;
                break;
            }
        }

        console.log(`Image ${img.name} matched to cow ID: ${matchedId}`);

        results.push({
            imageName: img.name,
            imageDate: img.date,
            cowId: matchedId,
            mimeType: "image/jpeg" 
        });
    }

    console.log("Final image → cow mapping:", results);
    return results;
}

function formatDateForName(date) {
    const pad = (n) => String(n).padStart(2, "0");
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
           `-${pad(date.getHours())}-${pad(date.getMinutes())}-${pad(date.getSeconds())}`;
}

/**
 * Downloads a file from Google Drive (or exports a Google app file) to disk.
 * @param {google.drive.Drive} drive - The Drive client.
 * @param {Object} file - The file metadata object.
 * @param {string} outPath - The local path to save the file.
 * @returns {Promise<string>} The output path upon completion.
 */
async function downloadToDisk(drive, file, outPath) {
    let apiCall;
    const mime = file.mimeType;
    console.log("MIME →", file.mimeType);

    try {
        // Use 'export' for Google native file types
        if (mime === "application/vnd.google-apps.spreadsheet") {
            apiCall = await drive.files.export(
                { fileId: file.id, mimeType: "text/csv" },
                { responseType: "stream" }
            );

        } else if (mime === "application/vnd.google-apps.document") {
            apiCall = await drive.files.export(
                { fileId: file.id, mimeType: "application/pdf" },
                { responseType: "stream" }
            );

        } else if (mime === "application/vnd.google-apps.presentation") {
            apiCall = await drive.files.export(
                { fileId: file.id, mimeType: "application/pdf" },
                { responseType: "stream" }
            );

        } else if (mime === "application/vnd.google-apps.drawing") {
            apiCall = await drive.files.export(
                { fileId: file.id, mimeType: "image/png" },
                { responseType: "stream" }
            );

        } else if (
            // Direct Download for non-native files (images, standard docs)
            mime === "image/jpeg" ||
            mime === "image/jpg" ||
            mime === "image/png" ||
            mime === "image/gif" ||
            mime === "image/webp"
        ) {
            // 🔥 DIRECT IMAGE DOWNLOAD (never use export!)
            apiCall = await drive.files.get(
                { fileId: file.id, alt: "media", supportsAllDrives: true },
                { responseType: "stream" }
            );

        } else {
            // Default: try direct download anyway
            apiCall = await drive.files.get(
                { fileId: file.id, alt: "media", supportsAllDrives: true },
                { responseType: "stream" }
            );
        }
    } catch (err) {
        console.error(`Error initiating export/download for file ${file.id}`, err);
        throw err;
    }

    // --------------------------------------------------------
    // FINAL: VALIDATE STREAM AND WRITE FILE
    // --------------------------------------------------------
    return new Promise((resolve, reject) => {
        // Check if the awaited call returned a valid stream object
        if (!apiCall?.data || typeof apiCall.data.pipe !== "function") {
            return reject(
                new Error(
                    `Download failed: fileId=${file.id}, no stream returned (mime=${mime})`
                )
            );
        }

        const dest = fs.createWriteStream(outPath);

        apiCall.data
            .on("error", (err) => { // Added error callback handling
                console.error(`Stream error during download of ${file.name}:`, err);
                reject(err);
            })
            .on("end", () => {
                console.log(`Successfully saved file to: ${outPath}`); // Added success log
                resolve(outPath);
            })
            .pipe(dest);
    });
}

/**
 * Downloads a file, specifically tailored for CSV/Spreadsheet/Excel sources.
 * @param {google.drive.Drive} drive - The Drive client.
 * @param {Object} file - The file metadata object.
 * @param {string} outPath - The local path to save the file.
 * @returns {Promise<string>} The output path upon completion.
 */
async function downloadToDiskCSV(drive, file, outPath) {
    let apiCall;
    const mime = file.mimeType;
    console.log("MIME →", mime);

    try {
        // --------------------------------------------------------
        // 1) GOOGLE SHEETS → EXPORT AS CSV
        // This is crucial for Google Sheets to be downloaded as CSV content.
        // --------------------------------------------------------
        if (mime === "application/vnd.google-apps.spreadsheet") {
            apiCall = await drive.files.export(
                { fileId: file.id, mimeType: "text/csv", supportsAllDrives: true },
                { responseType: "stream" }
            );
        }

        // --------------------------------------------------------
        // 2) EXCEL/CSV/IMAGE/ETC. FILES → DIRECT DOWNLOAD
        // --------------------------------------------------------
        else {
             // For any other file type (Excel, CSV, image, etc.), use direct media download.
            apiCall = await drive.files.get(
                { fileId: file.id, alt: "media", supportsAllDrives: true },
                { responseType: "stream" }
            );
        }


        // --------------------------------------------------------
        // FINAL: VALIDATE STREAM AND WRITE FILE
        // --------------------------------------------------------
        return new Promise((resolve, reject) => {
            if (!apiCall?.data || typeof apiCall.data.pipe !== "function") {
                return reject(
                    new Error(
                        `Download failed: fileId=${file.id}, no stream returned (mime=${mime})`
                    )
                );
            }

            const dest = fs.createWriteStream(outPath);

            apiCall.data
                .on("error", (err) => { // Added error callback handling
                    console.error(`Stream error during download of ${file.name}:`, err);
                    reject(err);
                })
                .on("end", () => {
                    console.log(`Successfully saved file to: ${outPath}`); // Added success log
                    resolve(outPath);
                })
                .pipe(dest);
        });

    } catch (err) {
        console.error(`Error downloading file ${file.id} (mime=${mime})`, err);
        throw err;
    }
}


function clearDownloadedImages() {
    const imagesDir = path.join(__dirname, "..", "utils", "downloads", "images");

    // Make sure directory exists
    if (!fs.existsSync(imagesDir)) {
        console.log("Images folder does not exist, nothing to delete.");
        return;
    }

    const files = fs.readdirSync(imagesDir);

    for (const file of files) {
        const filePath = path.join(imagesDir, file);
        try {
            fs.unlinkSync(filePath);
            console.log(`🗑 Deleted image: ${file}`);
        } catch (err) {
            console.error(`Failed to delete ${file}:`, err.message);
        }
    }

    console.log("✔ All images deleted.");
}

function extractTimestampFromFilename(filename) {
    // Handles:
    // 2025-07-18-07-53-00.jpg
    // 2025-07-15-07-50-44_cam0_cap4.jpg

    const cleanName = filename.split("_")[0]; // remove cam0_cap4 etc.
    const parts = cleanName.split("-");

    if (parts.length < 6) {
        throw new Error(`Invalid image filename format: ${filename}`);
    }

    const [year, month, day, hour, minute, second] = parts;

    return new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`);
}

// =============================================================
// IMAGE DOWNLOADER
// =============================================================

/**
 * Downloads a cow image file based on the first 4 digits of the cowId.
 * @param {string|number} cowId - The cow identifier.
 * @returns {Promise<Object>} Object containing file info and local path.
 */
async function downloadImage(cowId) {
    const drive = await getDriveClient();
    
    // --- 1. PREPARE PATHS AND CLEANUP ---
    clearDownloadedImages(); // Attempt to clear all previous images
    const shortId = String(cowId).slice(0, 4);

    // Find img folder
    const folderSearch = await drive.files.list({
        q: `'${FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and name='img' and trashed=false`,
        fields: "files(id, name)",
    });

    if (!folderSearch.data.files.length)
        throw new Error("Folder 'img' not found inside root folder.");

    const imgFolderId = folderSearch.data.files[0].id;

    // Search image by FIRST 4 digits only
    const fileSearch = await drive.files.list({
        q: `'${imgFolderId}' in parents and mimeType contains 'image/' and trashed = false and name contains '${shortId}'`,
        fields: "files(id, name, mimeType)",
    });

    if (!fileSearch.data.files.length)
        throw new Error(`Image starting with '${shortId}' not found inside /img folder.`);

    const file = fileSearch.data.files[0];
    const outPath = path.join(DOWNLOAD_DIR, "images", file.name);
    
    // Ensure the output directory exists
    fs.mkdirSync(path.dirname(outPath), { recursive: true });

    // -----------------------------------------------------------------
    // 🔥 NEW LOGIC: ABORT DOWNLOAD IF FILE STILL EXISTS AFTER CLEANUP
    // -----------------------------------------------------------------
    if (fs.existsSync(outPath)) {
        console.warn(`⚠️ Target image file ${file.name} still exists at ${outPath} after cleanup. Skipping download to avoid file lock crash.`);
        
        // Return the existing file's information
        return { id: file.id, name: file.name, localPath: outPath };
    }
    // -----------------------------------------------------------------

    console.log("Downloading image file:", file.name);

    // Use the corrected downloadToDisk function
    await downloadToDisk(drive, file, outPath);

    // 1. Extract timestamp
    const imgDate = extractTimestampFromFilename(file.name);

    // 2. Format new name
    const formatted = formatDateForName(imgDate);
    const newName = `${cowId}_${formatted}${path.extname(file.name)}`;
    const newPath = path.join(DOWNLOAD_DIR, "images", newName);

    // 3. Rename
    fs.renameSync(outPath, newPath);

    return { id: file.id, name: newName, localPath: newPath };

}


// =============================================================
// CSV INDIVIDUAL (FORCED CSV OUTPUT)
// =============================================================

/**
 * Downloads the individual CSV file for a specific cowId from the 'csvIndividual' folder.
 * @param {string|number} cowId - The cow identifier.
 * @returns {Promise<Object>} Object containing file info and local path.
 */
async function downloadCsv(cowId) {
    const drive = await getDriveClient();

    // Find 'csvIndividual' folder
    const folderSearch = await drive.files.list({
        // FIX: Consolidated query string
        q: `'${FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and name='csvIndividual' and trashed=false`,
        fields: "files(id, name)",
    });

    if (!folderSearch.data.files.length)
        throw new Error("Folder 'csvIndividual' not found.");

    const csvFolderId = folderSearch.data.files[0].id;

    // Search CSV file by cowId
    const fileSearch = await drive.files.list({
        // FIX: Consolidated query string
        q: `'${csvFolderId}' in parents and name contains '${cowId}' and trashed = false and (mimeType = 'text/csv' or mimeType = 'application/vnd.google-apps.spreadsheet' or mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')`,
        fields: "files(id, name, mimeType)",
    });

    if (!fileSearch.data.files.length)
        throw new Error(`CSV for cowId '${cowId}' not found.`);

    const file = fileSearch.data.files[0];

    // --- MODIFICATION: Ensure local file has a .csv extension ---
    const baseName = path.parse(file.name).name;
    const outPath = path.join(DOWNLOAD_DIR, "csv", `${baseName}.csv`);
    // ------------------------------------------------------------
    
    fs.mkdirSync(path.dirname(outPath), { recursive: true });

    // Use the specialized CSV downloader
    await downloadToDiskCSV(drive, file, outPath);

    return { id: file.id, name: file.name, localPath: outPath };
}



// =============================================================
// PATADAS — exactly ONE file (FORCED CSV OUTPUT)
// =============================================================

/**
 * Downloads the single 'patadas' CSV file.
 * @returns {Promise<Object>} Object containing file info and local path.
 */
async function downloadPatadas() {
    const drive = await getDriveClient();

    // Find 'patadas' folder
    const folderSearch = await drive.files.list({
        // FIX: Consolidated query string
        q: `'${FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and name='patadas' and trashed=false`,
        fields: "files(id, name)",
    });

    if (!folderSearch.data.files.length)
        throw new Error("Folder 'patadas' not found.");

    const folderId = folderSearch.data.files[0].id;

    // Search for CSV/Spreadsheet files in 'patadas'
    const fileSearch = await drive.files.list({
        // FIX: Consolidated query string
        // FIX APPLIED HERE: Include the folderId in parents to restrict the search.
        q: `'${folderId}' in parents and trashed=false and (mimeType='text/csv' or mimeType='application/vnd.google-apps.spreadsheet')`,
        fields: "files(id, name, mimeType)",
    });
    console.log(fileSearch.data.files);
    const files = fileSearch.data.files;

    if (files.length === 0)
        throw new Error("No CSV found in 'patadas' folder.");

    if (files.length > 1)
        throw new Error("More than one CSV in 'patadas'. Expected exactly one.");

    const file = files[0];

    // --- MODIFICATION: Ensure local file has a .csv extension ---
    const baseName = path.parse(file.name).name;
    const outPath = path.join(DOWNLOAD_DIR, "patadas", `${baseName}.csv`);
    // ------------------------------------------------------------

    fs.mkdirSync(path.dirname(outPath), { recursive: true });

    // Use the corrected downloadToDisk function (which is now downloadToDiskCSV for consistency)
    await downloadToDiskCSV(drive, file, outPath);

    return { id: file.id, name: file.name, localPath: outPath };
}



// =============================================================
// MAIN BUNDLE
// =============================================================

/**
 * Downloads all necessary files for a given cowId and computes the DEL.
 * @param {string|number} cowId - The cow identifier.
 * @returns {Promise<Object>} Object containing DEL calculation results and file paths.
 */
async function getCowDELBundle(cowId) {
    try {
        // 1. Download all required files concurrently (or sequentially if dependencies exist)
        const image = await downloadImage(cowId);
        const csv = await downloadCsv(cowId);
        const patadas = await downloadPatadas();

        // 2. Compute the DEL using the downloaded files
        const { ID, DEL } = await computeDELFromFiles(
            cowId,
            csv.localPath,
            patadas.localPath,
            image.localPath
        );

        // 3. Return the results
        return {
            cowId,
            ID,
            DEL,
            imagePath: image.localPath,
            csvPath: csv.localPath,
            patadasPath: patadas.localPath,
        };
    } catch (err) {
        console.error("getCowDELBundle error:", err);
        throw err;
    }
}



// --- Modified/Implemented Functions ---

/**
 * Reads the cow data file from the local path, parses it, and sorts it.
 * @param {string} filePath - The local path to the cow data CSV file.
 * @returns {Promise<CowDataRow[]>} An array of sorted cow data objects.
 */



// Export the primary functions
module.exports = {
    getDriveClient,
    getImageDates,
    getCowDELBundle,
    getFiles,
    getIdImag,
    downloadImage,
    downloadCsv,
};