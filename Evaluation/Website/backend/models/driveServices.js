const { google } = require("googleapis");
const { computeDELFromFiles } = require("./delService");
const path = require("path");
const fs = require("fs");
const KEY_FILE = "credentials.json";
const folderInfo = require("../../folder_info.json");

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

    // 1. Find the 'img' folder ID
    const folderSearch = await drive.files.list({
        // FIX: Consolidated query string to prevent newlines/spaces from breaking it
        q: `'${FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and name='img' and trashed=false`,
        fields: "files(id, name)",
    });

    if (!folderSearch.data.files.length) return [];

    const imgFolderId = folderSearch.data.files[0].id;

    // 2. List image files inside the 'img' folder
    const fileSearch = await drive.files.list({
        // FIX: Consolidated query string
        q: `'${imgFolderId}' in parents and mimeType contains 'image/' and trashed=false`,
        fields: "files(id, name, mimeType)",
    });

    return fileSearch.data.files;
}
// -------------------------------------------------------------


// =============================================================
// Helper — robust exporter/downloader for ANY Google Drive file
// =============================================================

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

    const shortId = String(cowId).slice(0, 4);

    // Find img folder
    const folderSearch = await drive.files.list({
        // FIX: Consolidated query string
        q: `'${FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and name='img' and trashed=false`,
        fields: "files(id, name)",
    });

    if (!folderSearch.data.files.length)
        throw new Error("Folder 'img' not found inside root folder.");

    const imgFolderId = folderSearch.data.files[0].id;

    // Search image by FIRST 4 digits only
    const fileSearch = await drive.files.list({
        // FIX: Consolidated query string
        q: `'${imgFolderId}' in parents and mimeType contains 'image/' and trashed = false and name contains '${shortId}'`,
        fields: "files(id, name, mimeType)",
    });

    if (!fileSearch.data.files.length)
        throw new Error(`Image starting with '${shortId}' not found inside /img folder.`);

    const file = fileSearch.data.files[0];
    console.log("Downloading image file:", file.name);
    const outPath = path.join(DOWNLOAD_DIR, "images", file.name);
    fs.mkdirSync(path.dirname(outPath), { recursive: true });

    // Use the corrected downloadToDisk function
    await downloadToDisk(drive, file, outPath);

    return { id: file.id, name: file.name, localPath: outPath };
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


// Export the primary functions
module.exports = {
    getDriveClient,
    getFiles,
    getCowDELBundle,
};