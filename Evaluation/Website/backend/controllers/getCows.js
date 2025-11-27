const driveService = require('../models/driveServices.js');
const path = require('path');


/**
 * Handles the main request to find, download, and process cow data. (Route /cow-data)
 */
exports.getCowData = async (req, res) => {
    try {
        const files = await driveService.getFiles();

        // Get all files whose filename begins with 4 digits and are images
        const cowFiles = files.filter(f =>
            /^[0-9]{4}/.test(f.name) && f.mimeType.startsWith("image/")
        );

        // Extract unique cow IDs
        const uniqueCowIds = [...new Set(
            cowFiles.map(f => f.name.substring(0, 4))
        )];

        res.render("cows", {   
            filesFound: cowFiles.length,
            cows: uniqueCowIds.map(id => ({ IDCow: id }))
        });

    } catch (error) {
        console.error("Error:", error);
        res.status(500).json({ error: error.message });
}
};