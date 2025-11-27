const pythonService = require("../utils/pythonService");
const driveService = require("../models/driveServices.js");
const { runProgram } = require("../utils/pythonService");
const path = require("path");

exports.getCowInfo = async (req, res) => {
    //const cowId = req.params.id;

    try {
        const dataset = path.join(__dirname, "..", "..", "dataset.json"); // path to your dataset
        const cowId = req.query.id || "123";  // example: get cow ID from query
        const output = path.join(__dirname, "..", "..", "output.png");   // where to save PNG

        const exeOutput = await runProgram(dataset, cowId, output);

        res.render("cowInfo", {
            cowID: 1234,
            bcs: 2.75,
            diasLeche: 143,
            estado: "Temporal",
            status: "rojo",
            graphImg: output
        });

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: "Failed to generate graph" });
    }

    //driveService.downloadImageAndCsv(cowId);

    // 🟩 Render page normally
    /**res.render("cowInfo", {
        cowID: 1234,
        bcs: 2.75,
        diasLeche: 143,
        estado: "Temporal",
        status: "rojo",
        graphImg: "/images/graph${cowId}.png"
    });*/
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
