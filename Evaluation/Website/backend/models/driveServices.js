const { google } = require("googleapis");
const fs = require('fs'); // Import File System module
const path = require('path'); // Import Path module
const KEY_FILE = "credentials.json"; // Your service account key file paths
const folderInfo = require("../../folder_info.json");
// Dynamic import is used for 'open' because it is an ES Module
// const open = require("open"); // <-- Removed due to ERR_REQUIRE_ESM
const  OAuth2Client  = google.auth.OAuth2;

// --- CONFIGURATION ---
const keys = {
    client_id: process.env.GOOGLE_CLIENT_ID,
    client_secret: process.env.GOOGLE_CLIENT_SECRET,
    redirect_uris: ["http://localhost:3001/oauth2callback"]
};


// --- CONFIGURATION ---
const FOLDER_ID = folderInfo.folder_id; // Google Drive Folder ID to read files from
const DOWNLOAD_DIR = path.join(__dirname, "..", "utils", "downloads"); // Local directory to save downloaded files
// Scopes required for reading files from a user's Drive
const SCOPE = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly"];




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
};
async function getFiles() {
    const drive = await getDriveClient();

    // 1. Find the subfolder named "img" INSIDE FOLDER_ID
    const folderSearch = await drive.files.list({
        q: `'${FOLDER_ID}' in parents and mimeType = 
        'application/vnd.google-apps.folder' and name = 'img' and trashed = false`,
        fields: "files(id, name)"
    });

    if (!folderSearch.data.files || folderSearch.data.files.length === 0) {
        console.log("img folder not found");
        return [];
    }

    const imgFolderId = folderSearch.data.files[0].id;
    console.log("Found img folder:", imgFolderId);

    // 2. Now search for images INSIDE the img folder
    const fileSearch = await drive.files.list({
        q: `'${imgFolderId}' in parents and mimeType contains 'image/' and trashed = false`,
        fields: "files(id, name, mimeType)"
    });

    return fileSearch.data.files;
}


/**
 * Downloads a file from Google Drive using its fileId
 */
async function downloadFile(drive, fileId, outputPath) {
    return new Promise(async (resolve, reject) => {
        try {
            const dest = fs.createWriteStream(outputPath);

            const response = await drive.files.get(
                { fileId, alt: "media" },
                { responseType: "stream" }
            );

            response.data
                .on("end", () => resolve(outputPath))
                .on("error", (err) => reject(err))
                .pipe(dest);

        } catch (err) {
            reject(err);
        }
    });
}

async function downloadImage(cowId) {
    const drive = await getDriveClient();

    // Find the "img" folder
    const folderSearch = await drive.files.list({
        q: `'${FOLDER_ID}' in parents and mimeType = 'application/vnd.google-apps.folder'
            and name = 'img' and trashed = false`,
        fields: "files(id, name)"
    });

    if (!folderSearch.data.files.length) {
        throw new Error(`Folder 'img' not found inside root folder.`);
    }

    const imgFolderId = folderSearch.data.files[0].id;

    // Now find an image starting with cowId
    const fileSearch = await drive.files.list({
        q: `'${imgFolderId}' in parents 
            and name contains '${cowId}' 
            and mimeType contains 'image/' 
            and trashed = false`,
        fields: "files(id, name, mimeType)"
    });

    if (!fileSearch.data.files.length) {
        throw new Error(`No image found for cowId '${cowId}'`);
    }

    const file = fileSearch.data.files[0];
    const outputPath = path.join(DOWNLOAD_DIR, file.name);

    await downloadFile(drive, file.id, outputPath);

    return {
        id: file.id,
        name: file.name,
        localPath: outputPath
    };
}

async function downloadCsv(cowId) {
    const drive = await getDriveClient();

    // Find the "csvIndividual" folder
    const folderSearch = await drive.files.list({
        q: `'${FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder'
            and name='csvIndividual' and trashed=false`,
        fields: "files(id, name)"
    });

    if (!folderSearch.data.files.length) {
        throw new Error(`Folder 'csvIndividual' not found.`);
    }

    const csvFolderId = folderSearch.data.files[0].id;

    // Find CSV matching cowId
    const fileSearch = await drive.files.list({
        q: `'${csvFolderId}' in parents 
            and name contains '${cowId}' 
            and name contains '.csv'
            and trashed = false`,
        fields: "files(id, name)"
    });

    if (!fileSearch.data.files.length) {
        throw new Error(`No CSV found for cowId '${cowId}'.`);
    }

    const csvFile = fileSearch.data.files[0];
    const outputPath = path.join(DOWNLOAD_DIR, csvFile.name);

    await downloadFile(drive, csvFile.id, outputPath);

    return {
        id: csvFile.id,
        name: csvFile.name,
        localPath: outputPath
    };
}

async function downloadPatadas() {
    const drive = await getDriveClient();

    // Find the "patadas" folder
    const folderSearch = await drive.files.list({
        q: `'${FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder'
            and name='patadas' and trashed=false`,
        fields: "files(id, name)"
    });

    if (!folderSearch.data.files.length) {
        throw new Error(`Folder 'patadas' not found.`);
    }

    const patadasFolderId = folderSearch.data.files[0].id;

    // Find CSV files inside patadas
    const fileSearch = await drive.files.list({
        q: `'${patadasFolderId}' in parents 
            and name contains '.csv'
            and trashed = false`,
        fields: "files(id, name)"
    });

    const files = fileSearch.data.files;

    if (files.length === 0) {
        throw new Error("No CSV found inside 'patadas' folder.");
    }

    if (files.length > 1) {
        throw new Error("More than one CSV found inside 'patadas'. Expected exactly 1.");
    }

    const csvFile = files[0];
    const outputPath = path.join(DOWNLOAD_DIR, csvFile.name);

    await downloadFile(drive, csvFile.id, outputPath);

    return {
        id: csvFile.id,
        name: csvFile.name,
        localPath: outputPath
    };
}


module.exports = {
    getDriveClient,
    getFiles,
    downloadImage,
    downloadCsv,
    downloadPatadas
};