const { google } = require("googleapis");
const url = require("url");
const fs = require('fs'); // Import File System module
const path = require('path'); // Import Path module
const KEY_FILE = "credentials.json"; // Your service account key file paths
const folderInfo = require("../../folder_info.json");
// Dynamic import is used for 'open' because it is an ES Module
// const open = require("open"); // <-- Removed due to ERR_REQUIRE_ESM
const  OAuth2Client  = google.auth.OAuth2;
// --- CONFIGURATION ---

// 💥 UPDATE THESE VALUES with your OAuth Client ID and Secret
const keys = {
    client_id: process.env.GOOGLE_CLIENT_ID,
    client_secret: process.env.GOOGLE_CLIENT_SECRET,
    redirect_uris: ["http://localhost:3001/oauth2callback"]
};


// --- CONFIGURATION ---
const FOLDER_ID = folderInfo.folder_id; // Google Drive Folder ID to read files from


// Scopes required for reading files from a user's Drive
const SCOPE = ["https://www.googleapis.com/auth/drive.readonly"];


const oauth2Client = new OAuth2Client(
    keys.client_id,
    keys.client_secret,
    keys.redirect_uris
);

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
    
    const query = `'${FOLDER_ID}' in parents and mimeType contains 'image/' and trashed = false`;
    
    const res = await drive.files.list({
        q: query,
        fields: "files(id, name, mimeType)"
    });

    return res.data.files;  // return files, do NOT send Express responses here
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

async function downloadImageAndCsv(imageFileId) {
    try {
        // Authenticate
        const auth = new google.auth.GoogleAuth({
            keyFile: KEY_FILE,
            scopes: SCOPE,
        });

        const client = await auth.getClient();
        const drive = google.drive({ version: "v3", auth: client });

        // -----------------------------------------------------
        // 1. Get metadata from the selected image
        // -----------------------------------------------------
        const imageMetadata = await drive.files.get({
            fileId: imageFileId,
            fields: "name"
        });

        const imageName = imageMetadata.data.name;
        const imagePrefix = imageName.slice(0, 4); // First 4 digits
        const imageLocalPath = path.join(DOWNLOAD_DIR, imageName);

        console.log(`Downloading image: ${imageName}`);

        await downloadFile(drive, imageFileId, imageLocalPath);

        // -----------------------------------------------------
        // 2. Search for CSV with matching 4-digit prefix
        // -----------------------------------------------------
        const csvQuery = `name contains '${imagePrefix}' and name contains '.csv'`;

        const csvSearch = await drive.files.list({
            q: csvQuery,
            fields: "files(id, name)",
        });

        if (!csvSearch.data.files.length) {
            throw new Error(`No CSV found starting with prefix '${imagePrefix}'`);
        }

        const csvFile = csvSearch.data.files[0];
        const csvLocalPath = path.join(DOWNLOAD_DIR, csvFile.name);

        console.log(`Downloading CSV: ${csvFile.name}`);

        await downloadFile(drive, csvFile.id, csvLocalPath);

        // -----------------------------------------------------
        // Done!
        // -----------------------------------------------------
        return {
            image: {
                fileId: imageFileId,
                name: imageName,
                path: imageLocalPath,
            },
            csv: {
                fileId: csvFile.id,
                name: csvFile.name,
                path: csvLocalPath,
            }
        };

    } catch (err) {
        console.error("Error in downloadImageAndCsv:", err);
        throw err;
    }
}

module.exports = {
    getDriveClient,
    getFiles,
    downloadImageAndCsv
};