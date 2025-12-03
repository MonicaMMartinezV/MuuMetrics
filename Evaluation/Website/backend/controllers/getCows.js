const driveService = require('../models/driveServices.js');
const path = require('path');

exports.getCowData = async (req, res) => {
    try {
        const files = await driveService.getIdImag();

        // Get all files whose filename begins with 4 digits and are images
        const cowFiles = files.filter(f =>
            /^[0-9]{4}/.test(f.cowId) && f.mimeType.startsWith("image/")
        );

        // Extract unique cow IDs
        //let uniqueCowIds = [...new Set(cowFiles.map(f => f.name.substring(0, 4)))];

        let uniqueCowIds = files.cowId;
        console.log("Unique Cow IDs:", cowFiles);

        if (cowFiles.length === 0) {
            // No IDs found → use fallback
            cows = [
                { IDCow: 1101 },
                { IDCow: 2105 },
                { IDCow: 3133 },
                { IDCow: 9199 }
            ];
        } else {
            // Use IDs from Drive
            cows = cowFiles.map(id => ({ IDCow: id.cowId }));
        }

        res.render("cows", {
            filesFound: cowFiles.length,
            cows,
            showError: false,
            showSuccess: false
        });

    } catch (error) {
        console.error("Error:", error);
        res.status(500).json({ error: error.message });
    }
};
