const driveService = require("../models/driveServices.js");
const { generateCowGraph } = require("../utils/graphService");

exports.getCowInfo = async (req, res) => {
    try {
        const cowId = req.params.cowId;

        //  Generate the graph (returns base64 data URI)
        const graphDataUri = await generateCowGraph(cowId);

        res.render("cowInfo", {
            cowID: cowId,
            bcs: 2.75,
            diasLeche: 143,
            estado: "Temporal",
            status: "rojo",
            graphImg: graphDataUri
        });

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: "Failed to generate graph" });
    }
};
exports.getCowById = async (req, res) => {
    try {
        const cowId = req.params.cowId;

        // Validate ID
        if (!/^[0-9]{4}$/.test(cowId)) {
            return res.status(400).send("Invalid Cow ID");
        }

        // Get Drive files from service
        const files = await driveService.getFiles();

        // Select files that match this cowId
        const cowFiles = files.filter(file =>
            file.name.startsWith(cowId)
        );

        if (cowFiles.length === 0) {
            return res.status(404).render("cowDetail", {
                cowId,
                message: "No files found for this cow",
                files: []
            });
        }

        res.render("cowDetail", {
            cowId,
            files: cowFiles,
            message: null
        });

    } catch (err) {
        console.error(err);
        res.status(500).send("Server error");
    }
};
